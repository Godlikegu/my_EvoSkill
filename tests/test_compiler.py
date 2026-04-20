import json

from myevoskill.compile_audit import HeuristicCompileAuditAdapter
from myevoskill.compiler import TaskBundleCompiler


def test_task_compiler_preserves_user_readme_notes_but_blocks_private_assets(tmp_path):
    source = tmp_path / "source_task"
    (source / "src").mkdir(parents=True)
    (source / "notebooks").mkdir(parents=True)
    (source / "plan").mkdir(parents=True)
    (source / "evaluation" / "reference_outputs").mkdir(parents=True)
    (source / "evaluation" / "tests").mkdir(parents=True)
    (source / "data").mkdir(parents=True)

    (source / "README.md").write_text(
        "\n".join(
            [
                "# Task",
                "keep this",
                "Implementation note: use scipy least_squares.",
                "Reference output note: compare against reference output semantics.",
                "Do not reveal data/ground_truth.npz.",
                "Do not reveal src/ or main.py.",
                "Do not reveal plan/design.md either.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (source / "main.py").write_text("print('private')\n", encoding="utf-8")
    (source / "src" / "solver.py").write_text("PRIVATE\n", encoding="utf-8")
    (source / "notebooks" / "demo.ipynb").write_text("{}", encoding="utf-8")
    (source / "plan" / "design.md").write_text("PRIVATE PLAN\n", encoding="utf-8")
    (source / "evaluation" / "reference_outputs" / "ref.npy").write_text(
        "ref", encoding="utf-8"
    )
    (source / "evaluation" / "tests" / "test_private.py").write_text(
        "assert False\n", encoding="utf-8"
    )
    (source / "data" / "raw_data.npz").write_text("raw", encoding="utf-8")
    (source / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    (source / "data" / "ground_truth.npz").write_text("gt", encoding="utf-8")

    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        source,
        task_id="task-1",
        family="optics",
        public_policy={
            "readme_policy": {
                "preserve_user_eval_notes": True,
                "remove_path_patterns": [
                    "(?i)data/ground_truth\\.npz",
                    "(?i)\\bsrc/",
                    "(?i)\\bmain\\.py\\b",
                    "(?i)\\bplan/",
                ],
            }
        },
    )

    public = bundle.public_bundle_dir
    hidden = bundle.hidden_bundle_dir

    assert (public / "README_public.md").exists()
    assert (public / "requirements.txt").exists()
    assert (public / "data" / "raw_data.npz").exists()
    assert (public / "data" / "meta_data.json").exists()
    assert not (public / "starter").exists()
    assert not (public / "main.py").exists()
    assert not (public / "src").exists()
    assert not (public / "notebooks").exists()
    assert not (public / "plan").exists()
    assert not (public / "evaluation").exists()
    assert not (public / "data" / "ground_truth.npz").exists()

    public_readme = (public / "README_public.md").read_text(encoding="utf-8")
    assert "Implementation note" in public_readme
    assert "Reference output note" in public_readme
    assert "ground_truth" not in public_readme
    assert "src/" not in public_readme
    assert "main.py" not in public_readme
    assert "plan/" not in public_readme

    assert (hidden / "main.py").exists()
    assert (hidden / "src" / "solver.py").exists()
    assert (hidden / "data" / "ground_truth.npz").exists()
    assert (hidden / "plan" / "design.md").exists()

    report = json.loads(bundle.compile_report_path.read_text(encoding="utf-8"))
    task_spec = json.loads(bundle.task_spec_path.read_text(encoding="utf-8"))
    assert "main.py" in report["blocked_paths"]
    assert "data/ground_truth.npz" in report["blocked_paths"]
    assert "plan/design.md" in report["rule_blocked_paths"]
    assert "data/raw_data.npz" in report["public_data_allowlist"]
    assert "ground_truth.npz" in report["public_data_denylist"]
    assert report["runtime_policy"]["model_timeout_seconds"] == 240
    assert report["runtime_policy"]["execution_budget_seconds"] == 900
    assert task_spec["runtime_policy"]["model_timeout_seconds"] == 240
    assert task_spec["runtime_policy"]["execution_budget_seconds"] == 900
    assert "task_contract_public" not in task_spec
    assert report["preserved_sections"]


def test_task_compiler_keeps_rule_barriers_even_with_conflicting_audit_suggestions(tmp_path):
    source = tmp_path / "source_task"
    (source / "src").mkdir(parents=True)
    (source / "data").mkdir(parents=True)
    (source / "README.md").write_text(
        "\n".join(
            [
                "# Problem Description",
                "Recover a field from data.",
                "# Method Hints",
                "- Use a stable inverse solver.",
                "Do not reveal src/private_solver.py or data/ground_truth.npz.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (source / "src" / "private_solver.py").write_text("private\n", encoding="utf-8")
    (source / "data" / "raw_data.npz").write_text("raw\n", encoding="utf-8")
    (source / "data" / "ground_truth.npz").write_text("gt\n", encoding="utf-8")

    class ConflictingAuditAdapter(HeuristicCompileAuditAdapter):
        def audit(self, readme_text, manifest, compile_context):
            result = super().audit(readme_text, manifest, compile_context)
            result["suggested_public_contract"] = {
                "required_outputs": [{"path": "src/private_solver.py"}],
                "judge_metrics": ["metric-a"],
            }
            return result

    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        source,
        task_id="task-2",
        family="physics",
        public_policy={
            "readme_policy": {
                "preserve_user_eval_notes": True,
                "remove_path_patterns": ["(?i)data/ground_truth\\.npz", "(?i)\\bsrc/"],
            },
            "public_data_allowlist": ["data/raw_data.npz"],
        },
        manifest={
            "proxy_output_name": "result.npz",
            "judge_metrics": ["metric-a"],
            "runtime_policy": {
                "model_timeout_seconds": 120,
                "execution_budget_seconds": 360,
            },
        },
        audit_adapter=ConflictingAuditAdapter(),
    )

    assert not (bundle.public_bundle_dir / "src").exists()
    report = json.loads(bundle.compile_report_path.read_text(encoding="utf-8"))
    task_spec = json.loads(bundle.task_spec_path.read_text(encoding="utf-8"))
    assert "src/private_solver.py" in report["rule_blocked_paths"]
    assert report["llm_suggested_public_contract"]["required_outputs"][0]["path"] == "src/private_solver.py"
    assert report["final_public_contract"]["required_outputs"][0]["path"] == "output/result.npz"
    assert report["runtime_policy"]["model_timeout_seconds"] == 120
    assert report["runtime_policy"]["execution_budget_seconds"] == 360
    assert task_spec["runtime_policy"]["model_timeout_seconds"] == 120
    assert task_spec["runtime_policy"]["execution_budget_seconds"] == 360
    assert "ignored_llm_contract_conflict_with_rule_barrier" in report["llm_audit_warnings"]


def test_task_compiler_removes_private_nested_readme_subsections(tmp_path):
    source = tmp_path / "source_task"
    (source / "data").mkdir(parents=True)
    (source / "README.md").write_text(
        "\n".join(
            [
                "# Data Description",
                "### data/raw_data.npz",
                "public data",
                "### data/ground_truth.npz",
                "private table row",
                "# Method Hints",
                "- keep this",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (source / "data" / "raw_data.npz").write_text("raw\n", encoding="utf-8")
    (source / "data" / "ground_truth.npz").write_text("gt\n", encoding="utf-8")

    bundle = TaskBundleCompiler(tmp_path / "compiled").compile(
        source,
        task_id="task-3",
        family="physics",
        public_policy={
            "readme_policy": {
                "remove_path_patterns": ["(?i)data/ground_truth\\.npz"],
            },
        },
    )
    public_readme = bundle.readme_public_path.read_text(encoding="utf-8")
    assert "data/raw_data.npz" in public_readme
    assert "data/ground_truth.npz" not in public_readme
    assert "private table row" not in public_readme
