from __future__ import annotations

import json
from pathlib import Path

from myevoskill.distill.episode_miner import mine_train_split
from myevoskill.distill.universe import DistillUniverse


def _repo_with_train_log(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "MyEvoSkill"
    run_root = repo / "artifacts" / "logs" / "Model" / "TrainTask"
    (run_root / "run-1").mkdir(parents=True)
    (run_root / "run-2").mkdir(parents=True)
    (tmp_path / "tasks" / "TrainTask").mkdir(parents=True)
    return repo, run_root


def _universe(repo: Path) -> DistillUniverse:
    return DistillUniverse(
        repo_root=repo,
        train_task_ids=("TrainTask",),
        valid_task_ids=("ValidTask",),
        model_slug="Model",
    )


def test_mine_train_split_skips_incomplete_newer_run(tmp_path: Path) -> None:
    repo, run_root = _repo_with_train_log(tmp_path)
    (run_root / "run-1" / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "verdict": "PASS",
                "rounds_used": 1,
                "runtime_seconds": 2.0,
                "policy": {"primary_output_rel": "output/reconstruction.npz"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run-1" / "trajectory.jsonl").write_text(
        '{"kind":"pre_tool_use","round_index":1,"tool_name":"Bash","tool_input":{"command":"python solve.py"}}\n',
        encoding="utf-8",
    )
    # Newer lexical run exists but is unfinished; it must not block mining.
    (run_root / "run-2" / "trajectory.jsonl").write_text(
        '{"kind":"pre_tool_use","tool_name":"Bash","tool_input":{"command":"python partial.py"}}\n',
        encoding="utf-8",
    )

    episodes = mine_train_split(_universe(repo))

    assert len(episodes) == 1
    assert episodes[0].run_id == "run-1"
    assert episodes[0].tool_uses[0].input_signature == "python solve.py"


def test_mine_train_split_accepts_summary_and_trajectory_json(tmp_path: Path) -> None:
    repo, run_root = _repo_with_train_log(tmp_path)
    (run_root / "run-1" / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "verdict": "PASS",
                "rounds_used": 1,
                "runtime_seconds": 2.0,
                "policy": {"primary_output_rel": "output/reconstruction.npz"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run-1" / "trajectory.json").write_text(
        json.dumps(
            [
                {
                    "kind": "pre_tool_use",
                    "round_index": 1,
                    "tool_name": "Bash",
                    "tool_input": {"command": "python solve_json.py"},
                }
            ]
        ),
        encoding="utf-8",
    )

    episodes = mine_train_split(_universe(repo))

    assert len(episodes) == 1
    assert episodes[0].tool_uses[0].input_signature == "python solve_json.py"


def test_mine_train_split_includes_recent_fail_timeout_and_rich_signals(tmp_path: Path) -> None:
    repo, run_root = _repo_with_train_log(tmp_path)
    for name, verdict in (("run-1", "FAIL"), ("run-2", "TIMEOUT"), ("run-3", "PASS")):
        run = run_root / name
        run.mkdir(exist_ok=True)
        (run / "run_summary.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "verdict": verdict,
                    "rounds_used": 1,
                    "runtime_seconds": 2.0,
                    "policy": {"primary_output_rel": "output/reconstruction.npz"},
                    "feedback_history": [
                        {
                            "round": 1,
                            "feedback": {
                                "verdict": "PASS" if verdict == "PASS" else "FAIL",
                                "failure_tags": [] if verdict == "PASS" else ["metric_below_threshold"],
                                "metric_status": {"ncc": verdict == "PASS"},
                            },
                        }
                    ],
                    "plan_history": [{"round": 1, "diff_lines": 3, "note": "ok"}],
                }
            ),
            encoding="utf-8",
        )
        (run / "judge_round_01.json").write_text(
            json.dumps({"judge_result": {"metrics_actual": {"ncc": 0.5}}}),
            encoding="utf-8",
        )
        (run / "trajectory.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "kind": "tool_call",
                            "round": 1,
                            "tool": "Bash",
                            "tool_use_id": f"{name}-tool",
                            "input": {"command": "python work/main.py --scale 0.01"},
                        }
                    ),
                    json.dumps(
                        {
                            "kind": "tool_result",
                            "round": 1,
                            "tool_use_id": f"{name}-tool",
                            "text": "loss_start=3.0\nloss_end=1.0\nCFL=0.4",
                            "is_error": False,
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        workspace = repo / "artifacts" / "workspaces" / "Model" / "TrainTask" / name / "work"
        workspace.mkdir(parents=True)
        (workspace / "main.py").write_text(
            "import subprocess, time\n"
            "lr = 0.01\n"
            "scale = 0.01\n"
            "t0 = time.time()\n"
            "subprocess.run(['python', '.claude/skills/name/scripts/helper.py', '--scale', str(scale)])\n",
            encoding="utf-8",
        )

    episodes = mine_train_split(_universe(repo))

    verdicts = [ep.final_verdict for ep in episodes]
    assert verdicts == ["PASS", "FAIL", "TIMEOUT"]
    failed = next(ep for ep in episodes if ep.final_verdict == "FAIL")
    assert failed.failure_signals[0].metric_status == {"ncc": False}
    assert failed.metrics_actual == {"ncc": 0.5}
    assert "CFL=0.4" in failed.tool_uses[0].result_tail
    assert failed.main_py_digest.has_timing_probe is True
    assert any("helper.py" in call for call in failed.main_py_digest.helper_calls)
    assert "scale=0.01" in failed.main_py_digest.hardcoded_constants
