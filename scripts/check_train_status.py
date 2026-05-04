"""Quick status report for the wave_optics_v1 train split.

Prints verdict / rounds_used for every run-* directory under
artifacts/logs/<model_slug>/<train_task>/. Used by the operator to
decide which train tasks still need a successful run before we can
distill a complete skill.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    log_root = repo_root / "artifacts" / "logs" / "Vendor2_Claude-4.6-opus"
    split_path = repo_root / "registry" / "splits" / "wave_optics_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    trains = split.get("train", [])
    valids = split.get("valid", [])

    def _scan(group: str, ids: list[str]) -> dict[str, str]:
        verdicts: dict[str, str] = {}
        print(f"\n=== {group} ({len(ids)}) ===")
        for tid in ids:
            tdir = log_root / tid
            best = "NO_RUN"
            if tdir.exists():
                runs = sorted(tdir.glob("run-*"))
                for rdir in runs:
                    sp = rdir / "run_summary.json"
                    if not sp.exists():
                        sp = rdir / "summary.json"

                    if not sp.exists():
                        continue
                    try:
                        s = json.loads(sp.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    v = str(s.get("verdict") or "?")
                    rounds = s.get("rounds_used", "?")
                    print(f"  {tid:35s} {rdir.name}  verdict={v:8s} rounds={rounds}")
                    if v == "PASS":
                        best = "PASS"
                    elif best != "PASS" and v in {"FAIL", "ERROR", "TIMEOUT"}:
                        best = v
                if not runs:
                    print(f"  {tid:35s} (no run-* dirs)")
            else:
                print(f"  {tid:35s} (no log dir)")
            verdicts[tid] = best
        return verdicts

    train_v = _scan("TRAIN", trains)
    valid_v = _scan("VALID", valids)

    train_pass = [t for t, v in train_v.items() if v == "PASS"]
    train_todo = [t for t, v in train_v.items() if v != "PASS"]
    print("\n--- summary ---")
    print(f"TRAIN passing : {len(train_pass)}/{len(trains)}  -> {train_pass}")
    print(f"TRAIN todo    : {train_todo}")
    print(f"VALID passing : {[t for t,v in valid_v.items() if v == 'PASS']}")
    print(f"VALID baseline-todo (no run yet): "
          f"{[t for t,v in valid_v.items() if v == 'NO_RUN']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
