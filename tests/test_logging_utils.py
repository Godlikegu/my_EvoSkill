import json

from myevoskill.logging_utils import RunLogger


def test_run_logger_persists_summary_and_logs(tmp_path):
    logger = RunLogger(tmp_path)
    run_dir = logger.create_run_dir("run-1")
    summary_path = logger.write_summary(run_dir, {"status": "validated", "count": 2})
    stdout_path = logger.append_text_log(run_dir, "stdout.log", "hello world")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "validated"
    assert stdout_path.read_text(encoding="utf-8").strip() == "hello world"


def test_run_logger_writes_json_artifact(tmp_path):
    logger = RunLogger(tmp_path)
    run_dir = logger.create_run_dir("run-2")
    path = logger.write_json_artifact(run_dir, "artifacts_manifest.json", {"a": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["a"] == 1
