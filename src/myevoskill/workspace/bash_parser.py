"""Static analysis of Bash commands to extract read/write file targets.

Used by ``harness/hooks.py`` to enforce write-permission and read-policy on
``Bash`` tool calls *before* the agent's command actually runs.

The parser is intentionally **conservative**: when in doubt, it tags the
target as ``dynamic`` (origin/path expression we cannot evaluate). The
hook layer treats any ``dynamic`` target as a hard deny, which forces the
agent to use the explicit ``Read`` / ``Write`` tools instead of hiding
filesystem access inside ``python -c``.

Public API
----------
``parse_bash_writes(command) -> BashAccess``
    Return a structured record of statically-detected reads / writes and
    a list of "dynamic" reasons that the caller should treat as deny.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass, field
from typing import List


# Python builtins / stdlib calls that *write* to a literal path argument.
# Each tuple is (callable_name, positional_index_of_path).
_WRITE_FUNCS_POS: dict[tuple[str, ...], int] = {
    # open(path, mode) -- mode handled separately
    ("open",): 0,
    ("os", "remove"): 0,
    ("os", "unlink"): 0,
    ("os", "rmdir"): 0,
    ("os", "mkdir"): 0,
    ("os", "makedirs"): 0,
    ("os", "rename"): 0,  # also a read of src; we treat both as writes
    ("os", "replace"): 0,
    ("shutil", "rmtree"): 0,
    ("shutil", "copy"): 1,   # dst at index 1
    ("shutil", "copy2"): 1,
    ("shutil", "copyfile"): 1,
    ("shutil", "move"): 1,
    # numpy / scipy / json
    ("np", "save"): 0,
    ("np", "savez"): 0,
    ("np", "savez_compressed"): 0,
    ("numpy", "save"): 0,
    ("numpy", "savez"): 0,
    ("numpy", "savez_compressed"): 0,
    ("json", "dump"): 1,  # dump(obj, fp) -- fp index 1; we still flag the open
}

# Read-only callables we want to surface so paths get the forbidden-substring
# / is_inside check (preventing reads of ground-truth or out-of-workspace
# data even when they only "read").
_READ_FUNCS_POS: dict[tuple[str, ...], int] = {
    ("open",): 0,  # only when mode is read; handled in walker
    ("np", "load"): 0,
    ("numpy", "load"): 0,
    ("json", "load"): 0,
}


# Path-method writes on a Path-like literal: Path("foo").write_text(...)
_PATH_WRITE_METHODS = {
    "write_text",
    "write_bytes",
    "touch",
    "unlink",
    "mkdir",
    "rename",
    "replace",
    "symlink_to",
}
_PATH_READ_METHODS = {
    "read_text",
    "read_bytes",
    "open",  # Path("x").open(...) -- mode determines read/write
}


@dataclass
class BashAccess:
    """Structured result of parsing one Bash command."""

    writes: List[str] = field(default_factory=list)
    reads: List[str] = field(default_factory=list)
    # Reasons the parser bailed out — caller MUST treat presence of any
    # dynamic entry as a hard deny.
    dynamic: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- #
# Top-level entry point
# ---------------------------------------------------------------------------- #


def parse_bash_writes(command: str) -> BashAccess:
    """Parse *command* and return statically-detected read/write paths.

    The function recognises:

    * shell redirections ``> dst``, ``>> dst``, ``2> dst``, ``|& tee dst``,
      ``tee [-a] dst``;
    * common write commands ``cp src dst``, ``mv src dst``, ``rm path``,
      ``mkdir path``, ``touch path``, ``chmod _ path`` (path is detected
      heuristically as the last positional arg);
    * Python one-liners and heredocs: ``python -c "<src>"`` and
      ``python -<<EOF ... EOF`` are AST-parsed, recognising literal-path
      writes through ``open(<lit>, 'w'|'a'|'x'|'r+'|...)``,
      ``Path(<lit>).write_text/...``, ``os.remove/replace/rename/...``,
      ``shutil.copy*/move/rmtree``, ``np.save*``, ``json.dump``;
    * ``python <script>`` is *not* parsed (we don't read script files);
      the script itself is reported as a read so the workspace boundary
      check applies.

    Anything we cannot statically evaluate (computed strings, glob
    expansions, command substitution, eval/exec, dynamic open() argument)
    is recorded as a ``dynamic`` reason.
    """

    access = BashAccess()
    if not command or not command.strip():
        return access

    # Block obvious eval/exec primitives outright.
    if re.search(r"(?<!\w)(eval|exec|source)\b", command):
        access.dynamic.append("uses eval/exec/source")
        return access

    # Detect command substitution and process substitution -- we cannot
    # statically evaluate these, so any path coming out of them is dynamic.
    if "$(" in command or "`" in command or "<(" in command or ">(" in command:
        access.dynamic.append("uses command/process substitution")

    # 1. Shell redirections.
    _scan_redirections(command, access)

    # 2. Tokenised pass for cp/mv/rm/mkdir/touch/chmod/tee + python -c / -<<.
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unbalanced quotes etc -- bail conservatively.
        access.dynamic.append("unparseable shell tokens")
        return access

    _scan_tokens(tokens, command, access)
    return access


# ---------------------------------------------------------------------------- #
# Shell redirection scanner
# ---------------------------------------------------------------------------- #


# Match ">", ">>", "2>", "&>" followed by a target. We don't try to be a full
# parser; we just look for redirection ops outside of quoted regions.
_REDIR_RE = re.compile(
    r"""
    (?<![<>&])              # not preceded by another redir char
    (?:&|\d)?               # optional fd (e.g. 2>)
    (>>|>)                  # the redir operator
    \s*
    (                       # the target
        "[^"]+"             # double-quoted
        | '[^']+'           # single-quoted
        | [^\s|;&<>]+       # bareword
    )
    """,
    re.VERBOSE,
)


def _scan_redirections(command: str, access: BashAccess) -> None:
    # Strip backslash-escaped redirects (rare in agent commands).
    for match in _REDIR_RE.finditer(command):
        target = match.group(2).strip().strip('"').strip("'")
        if not target:
            continue
        if "$" in target or "*" in target or "?" in target:
            access.dynamic.append(f"redir target uses expansion: {target}")
            continue
        access.writes.append(target)


# ---------------------------------------------------------------------------- #
# Token-level scanner (cp / mv / rm / tee / python -c)
# ---------------------------------------------------------------------------- #


_FILE_WRITE_CMDS = {"cp", "mv", "rm", "mkdir", "touch", "chmod", "chown", "ln"}


def _scan_tokens(tokens: List[str], full_command: str, access: BashAccess) -> None:
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        # cp / mv / rm / mkdir / touch / chmod / chown / ln: target = last
        # non-flag positional.
        if tok in _FILE_WRITE_CMDS:
            j = i + 1
            positionals: list[str] = []
            while j < n and tokens[j] not in {";", "&&", "||", "|"}:
                arg = tokens[j]
                if arg.startswith("-"):
                    j += 1
                    continue
                positionals.append(arg)
                j += 1
            if positionals:
                # For cp/mv/ln, the destination is the last positional;
                # source(s) are reads.
                if tok in {"cp", "mv", "ln"}:
                    if len(positionals) >= 2:
                        access.reads.extend(positionals[:-1])
                    access.writes.append(positionals[-1])
                else:
                    # rm / mkdir / touch / chmod / chown -- all positionals
                    # are write targets (they mutate the filesystem entry).
                    access.writes.extend(positionals)
            i = j
            continue

        # tee [-a] target [target...]
        if tok == "tee":
            j = i + 1
            while j < n and tokens[j] not in {";", "&&", "||", "|"}:
                arg = tokens[j]
                if arg in {"-a", "--append", "-i", "--ignore-interrupts"}:
                    j += 1
                    continue
                if arg.startswith("-"):
                    j += 1
                    continue
                access.writes.append(arg)
                j += 1
            i = j
            continue

        # python -c "<src>"
        if tok in {"python", "python3", "py"} and i + 2 < n and tokens[i + 1] == "-c":
            src = tokens[i + 2]
            _scan_python_source(src, access)
            i += 3
            continue

        # python <script> -- treat the script as a read.
        if (
            tok in {"python", "python3", "py"}
            and i + 1 < n
            and not tokens[i + 1].startswith("-")
        ):
            access.reads.append(tokens[i + 1])
            i += 2
            continue

        # python -<<EOF heredoc / python <<EOF -- captured heuristically by
        # locating the heredoc body in the *original* command string.
        if (
            tok in {"python", "python3", "py"}
            and i + 1 < n
            and tokens[i + 1].startswith("<<")
        ):
            body = _extract_heredoc(full_command, tokens[i + 1])
            if body is None:
                access.dynamic.append("python heredoc body could not be extracted")
            else:
                _scan_python_source(body, access)
            # Skip remaining tokens for this segment.
            while i < n and tokens[i] not in {";", "&&", "||"}:
                i += 1
            continue

        i += 1


def _extract_heredoc(command: str, op: str) -> str | None:
    """Pull out the body of ``<<EOF ... EOF`` from the raw command string."""

    m = re.search(r"<<-?\s*(['\"]?)(\w+)\1", command)
    if not m:
        return None
    delim = m.group(2)
    after = command[m.end():]
    end = re.search(rf"^\s*{re.escape(delim)}\s*$", after, re.MULTILINE)
    if not end:
        return None
    return after[: end.start()]


# ---------------------------------------------------------------------------- #
# Python source AST walker
# ---------------------------------------------------------------------------- #


def _scan_python_source(source: str, access: BashAccess) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        access.dynamic.append("python -c source is not valid Python")
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Call):
            _classify_call(node, access)


def _classify_call(call: ast.Call, access: BashAccess) -> None:
    qual = _qualified_name(call.func)
    args = call.args

    # 1. open(<lit>, mode)
    if qual == ("open",):
        path = _literal_str(args[0]) if args else None
        mode = _literal_str(args[1]) if len(args) > 1 else "r"
        if path is None:
            access.dynamic.append("open() with non-literal path")
            return
        if mode is None:
            mode = "r"
        if any(c in mode for c in ("w", "a", "x")) or "+" in mode:
            access.writes.append(path)
        else:
            access.reads.append(path)
        return

    # 2. Path(<lit>).<method>(...)
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and _qualified_name(call.func.value.func) in {("Path",), ("pathlib", "Path")}
    ):
        ctor_args = call.func.value.args
        path = _literal_str(ctor_args[0]) if ctor_args else None
        method = call.func.attr
        if path is None:
            access.dynamic.append(f"Path(...).{method}() with non-literal path")
            return
        if method in _PATH_WRITE_METHODS:
            access.writes.append(path)
        elif method in _PATH_READ_METHODS:
            # Path(...).open(<mode>) — inspect mode arg.
            if method == "open":
                mode = _literal_str(args[0]) if args else "r"
                if mode is not None and any(c in mode for c in ("w", "a", "x")) or (
                    mode and "+" in mode
                ):
                    access.writes.append(path)
                else:
                    access.reads.append(path)
            else:
                access.reads.append(path)
        return

    # 3. Module-level functions in our table.
    if qual in _WRITE_FUNCS_POS:
        idx = _WRITE_FUNCS_POS[qual]
        if idx < len(args):
            path = _literal_str(args[idx])
            if path is None:
                access.dynamic.append(f"{'.'.join(qual)}() with non-literal path")
            else:
                access.writes.append(path)
        return

    if qual in _READ_FUNCS_POS:
        idx = _READ_FUNCS_POS[qual]
        if idx < len(args):
            path = _literal_str(args[idx])
            if path is None:
                access.dynamic.append(f"{'.'.join(qual)}() with non-literal path")
            else:
                access.reads.append(path)
        return


def _qualified_name(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        if parent is None:
            return None
        return parent + (node.attr,)
    return None


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
