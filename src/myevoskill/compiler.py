"""Task compiler that creates public and hidden bundles."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .compile_audit import CompileAuditAdapter, NullCompileAuditAdapter
from .models import PublicExposurePolicy, READMEPolicy, TaskBundle
from .task_runtime import coerce_runtime_layout, coerce_runtime_policy
from .task_contract import PUBLIC_TASK_CONTRACT_FILENAME, load_task_contract


class TaskBundleCompiler:
    """Compile raw task assets into leakage-controlled bundles."""

    FORBIDDEN_PUBLIC_PATH_FRAGMENTS = (
        "/src/",
        "/notebooks/",
        "/plan/",
        "/evaluation/",
    )
    FORBIDDEN_PUBLIC_FILE_NAMES = {"main.py"}
    DEFAULT_PUBLIC_DATA_BLOCKLIST = {"ground_truth.npz", "baseline_reference.npz"}
    DEFAULT_REMOVE_PATTERNS = (
        r"(?i)hidden[_ ]judge",
        r"(?i)data/ground_truth\.npz",
        r"(?i)\bevaluation/",
        r"(?i)\bmain\.py\b",
        r"(?i)\bsrc/",
        r"(?i)\bnotebooks/",
        r"(?i)\bplan/",
    )

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def compile(
        self,
        source_task_dir: Path,
        task_id: str,
        family: str,
        public_policy: Optional[Dict[str, object]] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        audit_adapter: Optional[CompileAuditAdapter] = None,
    ) -> TaskBundle:
        source_task_dir = Path(source_task_dir)
        if not source_task_dir.exists():
            raise FileNotFoundError(f"Task source directory does not exist: {source_task_dir}")
        if not source_task_dir.is_dir():
            raise NotADirectoryError(f"Task source path is not a directory: {source_task_dir}")
        bundle_root = self.output_root / task_id
        public_root = bundle_root / "public_bundle"
        hidden_root = bundle_root / "hidden_bundle"
        policy = self._coerce_policy(public_policy)
        manifest = dict(manifest or {})
        audit_adapter = audit_adapter or NullCompileAuditAdapter()
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        public_root.mkdir(parents=True, exist_ok=True)
        hidden_root.mkdir(parents=True, exist_ok=True)

        copied_public: List[str] = []
        blocked_paths: List[str] = []
        sanitized_sections: List[str] = []
        preserved_sections: List[str] = []
        public_allowlist: List[str] = list(policy.public_data_allowlist)
        public_denylist: List[str] = sorted(
            set(self.DEFAULT_PUBLIC_DATA_BLOCKLIST) | set(policy.public_data_denylist)
        )
        readme_text = ""
        readme_path = source_task_dir / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8")
        audit_result = audit_adapter.audit(
            readme_text=readme_text,
            manifest=manifest,
            compile_context=self._build_compile_context(
                source_task_dir=source_task_dir,
                task_id=task_id,
                family=family,
                policy=policy,
            ),
        )
        llm_audit_warnings = list(audit_result.get("sanitization_warnings", []))
        llm_suggested_public_contract = audit_result.get("suggested_public_contract", {})

        self._copy_hidden_bundle(source_task_dir, hidden_root)

        public_contract_payload: Dict[str, Any] = {}
        try:
            public_contract_payload = load_task_contract(source_task_dir, public=True)
        except FileNotFoundError:
            public_contract_payload = {}
        if public_contract_payload:
            (public_root / PUBLIC_TASK_CONTRACT_FILENAME).write_text(
                json.dumps(public_contract_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            copied_public.append(PUBLIC_TASK_CONTRACT_FILENAME)
        declared_public_paths = {
            str(item.get("path", "") or "").replace("\\", "/").strip()
            for item in public_contract_payload.get("files", []) or []
            if isinstance(item, Mapping) and str(item.get("path", "") or "").strip()
        }

        for path in source_task_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source_task_dir)
            rel_str = rel.as_posix()
            if rel.parts[:2] == ("evaluation", PUBLIC_TASK_CONTRACT_FILENAME):
                continue
            if public_contract_payload and rel_str not in declared_public_paths:
                blocked_paths.append(rel_str)
                continue
            if not public_contract_payload and self._is_forbidden_public(rel):
                blocked_paths.append(rel_str)
                continue
            if rel.parts[:1] == ("data",) and rel.name in public_denylist:
                blocked_paths.append(rel_str)
                continue
            if rel.name == "README.md":
                target = public_root / "README_public.md"
                target.parent.mkdir(parents=True, exist_ok=True)
                sanitized_content, removed_sections, kept_sections = self._sanitize_readme(
                    path.read_text(encoding="utf-8"),
                    policy.readme_policy,
                )
                target.write_text(sanitized_content, encoding="utf-8")
                copied_public.append("README_public.md")
                sanitized_sections.extend(removed_sections)
                preserved_sections.extend(kept_sections)
                continue
            if rel.parts[:2] == ("evaluation", "reference_outputs"):
                blocked_paths.append(rel_str)
                continue
            target = public_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied_public.append(rel_str)
            if rel.parts[:1] == ("data",):
                public_allowlist.append(rel_str)

        policy_payload = self._serialize_policy(policy)
        task_spec = {
            "task_id": task_id,
            "family": family,
            "public_bundle": "public_bundle",
            "hidden_bundle": "hidden_bundle",
            "public_policy": policy_payload,
            "runtime_layout": coerce_runtime_layout(manifest.get("runtime_layout")),
            "runtime_policy": coerce_runtime_policy(manifest.get("runtime_policy")),
            "runtime_env": dict(manifest.get("runtime_env") or {}),
            "public_eval_spec": dict(manifest.get("public_eval_spec") or {}),
            "primary_output_path": str(manifest.get("primary_output_path") or ""),
            "task_contract_public_path": PUBLIC_TASK_CONTRACT_FILENAME,
            "task_contract_path": str(manifest.get("task_contract_path") or ""),
            "judge_adapter_path": str(manifest.get("judge_adapter_path") or ""),
            "ready": bool(manifest.get("ready", False)),
        }
        task_spec_path = bundle_root / "task.yaml"
        task_spec_path.write_text(
            json.dumps(task_spec, indent=2, sort_keys=True), encoding="utf-8"
        )

        final_public_contract, contract_warnings = self._finalize_public_contract(
            manifest=manifest,
            public_contract=public_contract_payload,
            audit_result=audit_result,
        )
        llm_audit_warnings.extend(contract_warnings)
        compile_report = {
            "task_id": task_id,
            "family": family,
            "copied_public": sorted(copied_public),
            "blocked_paths": sorted(set(blocked_paths)),
            "rule_blocked_paths": sorted(set(blocked_paths)),
            "preserved_sections": sorted(set(preserved_sections)),
            "sanitized_sections": sorted(set(sanitized_sections)),
            "public_data_allowlist": sorted(set(public_allowlist)),
            "public_data_denylist": sorted(set(public_denylist)),
            "llm_audit_warnings": sorted(set(llm_audit_warnings)),
            "llm_suggested_public_contract": llm_suggested_public_contract,
            "final_public_contract": final_public_contract,
            "runtime_layout": task_spec["runtime_layout"],
            "runtime_policy": task_spec["runtime_policy"],
            "runtime_env": task_spec["runtime_env"],
            "public_eval_spec": task_spec["public_eval_spec"],
            "primary_output_path": task_spec["primary_output_path"],
            "task_contract_public_path": task_spec["task_contract_public_path"],
            "task_contract_path": task_spec["task_contract_path"],
            "judge_adapter_path": task_spec["judge_adapter_path"],
            "ready": task_spec["ready"],
        }
        compile_report_path = bundle_root / "compile_report.json"
        compile_report_path.write_text(
            json.dumps(compile_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return TaskBundle(
            task_id=task_id,
            family=family,
            root_dir=bundle_root,
            public_bundle_dir=public_root,
            hidden_bundle_dir=hidden_root,
            task_spec_path=task_spec_path,
            compile_report_path=compile_report_path,
            readme_public_path=public_root / "README_public.md",
        )

    def compile_task(self, source_task_dir: Path, task_id: str, family: str) -> TaskBundle:
        """Backward-compatible wrapper."""
        return self.compile(source_task_dir=source_task_dir, task_id=task_id, family=family)

    def _copy_hidden_bundle(self, source_task_dir: Path, hidden_root: Path) -> None:
        for path in source_task_dir.rglob("*"):
            rel = path.relative_to(source_task_dir)
            target = hidden_root / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _is_forbidden_public(self, rel: Path) -> bool:
        rel_str = "/" + rel.as_posix().strip("/") + "/"
        if rel.name in self.FORBIDDEN_PUBLIC_FILE_NAMES:
            return True
        return any(fragment in rel_str for fragment in self.FORBIDDEN_PUBLIC_PATH_FRAGMENTS)

    def _sanitize_readme(
        self, content: str, policy: READMEPolicy
    ) -> Tuple[str, List[str], List[str]]:
        lines = content.splitlines()
        safe_lines: List[str] = []
        removed_sections: List[str] = []
        kept_sections: List[str] = []
        current_heading = ""
        skip_section = False
        patterns = list(self.DEFAULT_REMOVE_PATTERNS) + list(policy.remove_path_patterns)

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if stripped.startswith("#"):
                current_heading = stripped.lstrip("#").strip()
                heading_lower = current_heading.lower()
                if any(re.search(pattern, stripped) for pattern in patterns) or any(
                    re.search(pattern, current_heading) for pattern in patterns
                ):
                    skip_section = True
                    removed_sections.append(current_heading or "<unnamed section>")
                    continue
                if any(item.lower() == heading_lower for item in policy.preserve_sections):
                    skip_section = False
                    kept_sections.append(current_heading)
                    safe_lines.append(line)
                    continue
                if any(item.lower() == heading_lower for item in policy.remove_sections):
                    skip_section = True
                    removed_sections.append(current_heading or "<unnamed section>")
                    continue
                skip_section = False
                kept_sections.append(current_heading or "<root>")
                safe_lines.append(line)
                continue
            if skip_section:
                continue
            if not policy.preserve_user_eval_notes and any(
                token in lower for token in ("reference output", "implementation", "metrics")
            ):
                if current_heading:
                    removed_sections.append(current_heading)
                continue
            if any(re.search(pattern, line) for pattern in patterns):
                if current_heading:
                    removed_sections.append(current_heading)
                continue
            safe_lines.append(line)

        compact_lines: List[str] = []
        previous_blank = False
        for line in safe_lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compact_lines.append(line)
            previous_blank = is_blank
        compact_lines.append("")
        compact_lines.append(
            "Note: private asset paths and hidden evaluator internals are excluded."
        )
        return "\n".join(compact_lines).strip() + "\n", removed_sections, kept_sections

    def _coerce_policy(self, public_policy: Optional[Dict[str, object]]) -> PublicExposurePolicy:
        policy = dict(public_policy or {})
        readme_policy_dict = dict(policy.get("readme_policy", {}))
        readme_policy = READMEPolicy(
            preserve_sections=tuple(readme_policy_dict.get("preserve_sections", [])),
            remove_sections=tuple(readme_policy_dict.get("remove_sections", [])),
            remove_path_patterns=tuple(readme_policy_dict.get("remove_path_patterns", [])),
            preserve_user_eval_notes=readme_policy_dict.get(
                "preserve_user_eval_notes", True
            ),
        )
        return PublicExposurePolicy(
            readme_policy=readme_policy,
            public_data_allowlist=tuple(policy.get("public_data_allowlist", [])),
            public_data_denylist=tuple(policy.get("public_data_denylist", [])),
        )

    def _serialize_policy(self, policy: PublicExposurePolicy) -> Dict[str, object]:
        return {
            "readme_policy": {
                "preserve_sections": list(policy.readme_policy.preserve_sections),
                "remove_sections": list(policy.readme_policy.remove_sections),
                "remove_path_patterns": list(policy.readme_policy.remove_path_patterns),
                "preserve_user_eval_notes": policy.readme_policy.preserve_user_eval_notes,
            },
            "public_data_allowlist": list(policy.public_data_allowlist),
            "public_data_denylist": list(policy.public_data_denylist),
        }

    def _build_compile_context(
        self,
        source_task_dir: Path,
        task_id: str,
        family: str,
        policy: PublicExposurePolicy,
    ) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "family": family,
            "source_files": sorted(
                path.relative_to(source_task_dir).as_posix()
                for path in source_task_dir.rglob("*")
                if path.is_file()
            ),
            "forbidden_public_path_fragments": list(self.FORBIDDEN_PUBLIC_PATH_FRAGMENTS),
            "forbidden_public_file_names": sorted(self.FORBIDDEN_PUBLIC_FILE_NAMES),
            "public_policy": self._serialize_policy(policy),
        }

    def _finalize_public_contract(
        self,
        manifest: Mapping[str, Any],
        public_contract: Mapping[str, Any],
        audit_result: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        final_contract: Dict[str, Any] = {}
        output = dict(public_contract.get("output") or {})
        if output:
            final_contract["required_outputs"] = [
                {
                    "path": str(output.get("path", "") or ""),
                    "format": str(output.get("format", "") or ""),
                    "fields": [dict(item or {}) for item in output.get("fields", []) or []],
                }
            ]
        else:
            output_contract = dict(manifest.get("output_contract") or {})
            required_outputs = [dict(item or {}) for item in output_contract.get("required_outputs", []) or []]
            if required_outputs:
                final_contract["required_outputs"] = required_outputs
            else:
                proxy_output_name = str(manifest.get("proxy_output_name", "") or "").strip()
                if proxy_output_name:
                    final_contract["required_outputs"] = [
                        {
                            "path": proxy_output_name
                            if "/" in proxy_output_name
                            else f"output/{proxy_output_name}",
                            "format": "npz",
                        }
                    ]
        metrics = [
            str(item.get("name", "") or "")
            for item in public_contract.get("metrics", []) or []
            if isinstance(item, Mapping) and str(item.get("name", "") or "")
        ]
        if not metrics:
            metrics = [str(item) for item in manifest.get("judge_metrics", []) or [] if str(item)]
        if metrics:
            final_contract["judge_metrics"] = metrics

        suggested_contract = dict(audit_result.get("suggested_public_contract", {}))
        suggested_method_hints = list(audit_result.get("suggested_method_hints", []))
        safe_method_hints: List[str] = []
        for hint in suggested_method_hints:
            if self._contains_forbidden_reference(str(hint)):
                warnings.append(f"ignored_llm_method_hint:{hint}")
                continue
            safe_method_hints.append(str(hint))
        if safe_method_hints:
            final_contract["method_hints"] = safe_method_hints

        if suggested_contract:
            serialized = json.dumps(suggested_contract, sort_keys=True, default=str)
            if self._contains_forbidden_reference(serialized):
                warnings.append("ignored_llm_contract_conflict_with_rule_barrier")
            else:
                for key, value in suggested_contract.items():
                    final_contract.setdefault(key, value)
        return final_contract, warnings

    def _contains_forbidden_reference(self, text: str) -> bool:
        normalized = text.replace("\\", "/")
        lowered = normalized.lower()
        path_fragments = (
            "src/",
            "main.py",
            "notebooks/",
            "plan/",
            "evaluation/tests/",
            "evaluation/reference_outputs/",
            "ground_truth",
        )
        if any(fragment in lowered for fragment in path_fragments):
            return True
        return any(re.search(pattern, normalized) for pattern in self.DEFAULT_REMOVE_PATTERNS)


class TaskCompiler(TaskBundleCompiler):
    """Backward-compatible compiler name."""
