from pathlib import Path

import numpy as np
import pytest

from myevoskill.task_contract import (
    derive_public_task_contract,
    evaluate_metric,
    validate_output_payload_against_contract,
    validate_task_contract,
    validate_task_contract_shapes,
    validate_task_contract_task_paths,
)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_validate_output_payload_accepts_finite_scalar_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "reconstruction.npz"
    _write_npz(output_path, scalar=np.float64(1.25))
    contract = {
        "output": {
            "fields": [
                {
                    "name": "scalar",
                    "dtype": "float64",
                    "shape": [],
                }
            ]
        }
    }

    with np.load(output_path, allow_pickle=False) as payload:
        result = validate_output_payload_against_contract(payload, contract)

    assert result["missing_fields"] == []
    assert result["warnings"] == []


def test_validate_output_payload_flags_scalar_nan(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "reconstruction.npz"
    _write_npz(output_path, scalar=np.float64(np.nan))
    contract = {
        "output": {
            "fields": [
                {
                    "name": "scalar",
                    "dtype": "float64",
                    "shape": [],
                }
            ]
        }
    }

    with np.load(output_path, allow_pickle=False) as payload:
        result = validate_output_payload_against_contract(payload, contract)

    assert result["warnings"] == ["nan_or_inf field: scalar"]


def test_evaluate_metric_supports_python_callable_kwargs_mapping_key(tmp_path):
    helper_path = tmp_path / "src" / "metric_helper.py"
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(
        "\n".join(
            [
                "def compute_metrics(estimate, reference):",
                "    return {'score': float(estimate.sum() + reference.sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_npz(tmp_path / "data" / "ground_truth.npz", image=np.array([[1.0, 2.0]], dtype=np.float32))
    output_path = tmp_path / "output" / "reconstruction.npz"
    _write_npz(output_path, reconstruction=np.array([[3.0, 4.0]], dtype=np.float32))

    contract = {
        "version": 2,
        "task_id": "demo_metric_task",
        "family": "demo",
        "files": [
            {
                "id": "ground_truth",
                "path": "data/ground_truth.npz",
                "visibility": "private",
                "role": "reference_data",
                "semantics": "Reference output.",
                "fields": {"image": {"dtype": "float32", "shape": [1, 2]}},
            },
            {
                "id": "metric_helper",
                "path": "src/metric_helper.py",
                "visibility": "private",
                "role": "metric_helper",
                "semantics": "Metric helper.",
            },
        ],
        "execution": {
            "read_first": [],
            "readable_files": [],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "reconstruction",
                    "dtype": "float32",
                    "shape": [1, 2],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "score",
                "goal": "maximize",
                "threshold": 0.0,
                "helper": {
                    "interface": "python_callable",
                    "file_id": "metric_helper",
                    "callable": "compute_metrics",
                    "invocation": {"mode": "kwargs"},
                    "result": {"mode": "mapping_key", "key": "score"},
                },
                "inputs": {
                    "estimate": {
                        "source": "output",
                        "field": "reconstruction",
                        "selectors": {"index": 0},
                        "expected_shape": [2],
                    },
                    "reference": {
                        "source": "file",
                        "file_id": "ground_truth",
                        "field": "image",
                        "selectors": {"index": 0},
                        "expected_shape": [2],
                    },
                },
            }
        ],
    }

    assert validate_task_contract(contract) == []
    assert validate_task_contract_task_paths(tmp_path, contract) == []
    assert validate_task_contract_shapes(tmp_path, contract) == []

    with np.load(output_path, allow_pickle=False) as output_payload:
        value = evaluate_metric(tmp_path, contract, contract["metrics"][0], output_payload=output_payload)

    assert value == 10.0


def test_evaluate_metric_supports_python_callable_mapping_scalar(tmp_path):
    helper_path = tmp_path / "src" / "metric_helper.py"
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(
        "\n".join(
            [
                "def compute_metric(inputs):",
                "    return float(inputs['estimate'].sum())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output" / "reconstruction.npz"
    _write_npz(output_path, reconstruction=np.array([[3.0, 4.0]], dtype=np.float32))

    contract = {
        "version": 2,
        "task_id": "demo_mapping_task",
        "family": "demo",
        "files": [
            {
                "id": "metric_helper",
                "path": "src/metric_helper.py",
                "visibility": "private",
                "role": "metric_helper",
                "semantics": "Metric helper.",
            }
        ],
        "execution": {
            "read_first": [],
            "readable_files": [],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "reconstruction",
                    "dtype": "float32",
                    "shape": [1, 2],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "estimate_sum",
                "goal": "maximize",
                "threshold": 0.0,
                "helper": {
                    "interface": "python_callable",
                    "file_id": "metric_helper",
                    "callable": "compute_metric",
                    "invocation": {"mode": "mapping"},
                    "result": {"mode": "scalar"},
                },
                "inputs": {
                    "estimate": {
                        "source": "output",
                        "field": "reconstruction",
                        "selectors": {"index": 0},
                        "expected_shape": [2],
                    }
                },
            }
        ],
    }

    assert validate_task_contract(contract) == []
    assert validate_task_contract_task_paths(tmp_path, contract) == []
    assert validate_task_contract_shapes(tmp_path, contract) == []

    with np.load(output_path, allow_pickle=False) as output_payload:
        value = evaluate_metric(tmp_path, contract, contract["metrics"][0], output_payload=output_payload)

    assert value == 7.0


def test_derive_public_task_contract_preserves_nested_value_bindings():
    contract = {
        "version": 2,
        "task_id": "demo_public_contract",
        "family": "demo",
        "files": [
            {
                "id": "raw_data",
                "path": "data/raw_data.npz",
                "visibility": "public",
                "role": "input_data",
                "semantics": "Input data.",
                "fields": {"signal": {"dtype": "float32", "shape": [1, 2]}},
            },
            {
                "id": "ground_truth",
                "path": "data/ground_truth.npz",
                "visibility": "private",
                "role": "reference_data",
                "semantics": "Hidden reference.",
                "fields": {"label": {"dtype": "float32", "shape": [1]}},
            },
        ],
        "execution": {
            "read_first": ["raw_data"],
            "readable_files": ["raw_data"],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "estimated_temperature_K",
                    "dtype": "float32",
                    "shape": [1],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "temperature_error",
                "goal": "minimize",
                "threshold": 1.0,
                "helper": {"interface": "builtin", "builtin": "nrmse"},
                "inputs": {
                    "params_pred": {
                        "source": "value",
                        "value": {
                            "temperature": {
                                "source": "output",
                                "field": "estimated_temperature_K",
                                "selectors": {"index": 0},
                                "expected_shape": [],
                            }
                        },
                        "expected_shape": [1],
                    },
                    "params_true": {
                        "source": "value",
                        "value": {
                            "temperature": {
                                "source": "file",
                                "file_id": "ground_truth",
                                "field": "label",
                                "selectors": {"index": 0},
                                "expected_shape": [],
                            }
                        },
                        "expected_shape": [1],
                    },
                },
            }
        ],
    }

    public_contract = derive_public_task_contract(contract)
    params_pred = public_contract["metrics"][0]["inputs"]["params_pred"]
    params_true = public_contract["metrics"][0]["inputs"]["params_true"]

    assert params_pred["value"]["temperature"]["source"] == "output"
    assert params_pred["value"]["temperature"]["field"] == "estimated_temperature_K"
    assert params_true["value"]["temperature"]["source"] == "file"
    assert params_true["value"]["temperature"]["file_id"] == "ground_truth"
    assert params_true["value"]["temperature"]["visibility"] == "private"


def test_validate_task_contract_rejects_invalid_explicit_helper_input_schema():
    contract = {
        "version": 2,
        "task_id": "demo_invalid_helper_schema",
        "family": "demo",
        "files": [
            {
                "id": "metric_helper",
                "path": "src/metric_helper.py",
                "visibility": "private",
                "role": "metric_helper",
                "semantics": "Metric helper.",
            }
        ],
        "execution": {
            "read_first": [],
            "readable_files": [],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "reconstruction",
                    "dtype": "float32",
                    "shape": [2],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "score",
                "goal": "maximize",
                "threshold": 0.0,
                "helper": {
                    "interface": "python_callable",
                    "file_id": "metric_helper",
                    "callable": "compute_metrics",
                    "invocation": {"mode": "kwargs"},
                    "result": {"mode": "scalar"},
                    "input_schema": {
                        "required": ["estimate", "reference"],
                        "optional": ["estimate"],
                    },
                },
                "inputs": {
                    "estimate": {
                        "source": "output",
                        "field": "reconstruction",
                        "expected_shape": [2],
                    },
                    "reference": {
                        "source": "value",
                        "value": [1.0, 2.0],
                        "expected_shape": [2],
                    },
                },
            }
        ],
    }

    errors = validate_task_contract(contract)

    assert any("helper.input_schema is invalid" in error for error in errors)


def test_validate_task_contract_task_paths_rejects_metric_adapter_input_binding_name_mismatch(
    tmp_path,
):
    helper_path = tmp_path / "evaluation" / "task_metric_adapter.py"
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(
        "\n".join(
            [
                "METRIC_RECIPES = {'nrmse': {'op': 'nrmse'}}",
                "",
                "def compute_metric(metric_name, inputs):",
                "    return 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    contract = {
        "version": 2,
        "task_id": "demo_metric_adapter_schema",
        "family": "demo",
        "files": [
            {
                "id": "metric_adapter",
                "path": "evaluation/task_metric_adapter.py",
                "visibility": "private",
                "role": "metric_helper",
                "semantics": "Metric adapter.",
            }
        ],
        "execution": {
            "read_first": [],
            "readable_files": [],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "reconstruction",
                    "dtype": "float32",
                    "shape": [2],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "nrmse",
                "goal": "minimize",
                "threshold": 1.0,
                "helper": {
                    "interface": "metric_adapter",
                    "file_id": "metric_adapter",
                    "callable": "compute_metric",
                },
                "inputs": {
                    "image": {
                        "source": "value",
                        "value": [1.0, 2.0],
                        "expected_shape": [2],
                    },
                    "reference": {
                        "source": "value",
                        "value": [1.0, 2.0],
                        "expected_shape": [2],
                    },
                },
            }
        ],
    }

    assert validate_task_contract(contract) == []
    errors = validate_task_contract_task_paths(tmp_path, contract)

    assert any("missing required helper input bindings" in error for error in errors)
    assert any("unexpected helper input bindings" in error for error in errors)


def test_evaluate_metric_rejects_metric_adapter_input_binding_name_mismatch(tmp_path):
    helper_path = tmp_path / "evaluation" / "task_metric_adapter.py"
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(
        "\n".join(
            [
                "METRIC_RECIPES = {'nrmse': {'op': 'nrmse'}}",
                "",
                "def compute_metric(metric_name, inputs):",
                "    raise AssertionError('should not execute helper when bindings are invalid')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    contract = {
        "version": 2,
        "task_id": "demo_runtime_metric_adapter_schema",
        "family": "demo",
        "files": [
            {
                "id": "metric_adapter",
                "path": "evaluation/task_metric_adapter.py",
                "visibility": "private",
                "role": "metric_helper",
                "semantics": "Metric adapter.",
            }
        ],
        "execution": {
            "read_first": [],
            "readable_files": [],
            "entrypoint": "work/main.py",
            "writable_paths": ["output/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [
                {
                    "name": "reconstruction",
                    "dtype": "float32",
                    "shape": [2],
                    "semantics": "Prediction.",
                }
            ],
        },
        "metrics": [
            {
                "name": "nrmse",
                "goal": "minimize",
                "threshold": 1.0,
                "helper": {
                    "interface": "metric_adapter",
                    "file_id": "metric_adapter",
                    "callable": "compute_metric",
                },
                "inputs": {
                    "image": {
                        "source": "value",
                        "value": [1.0, 2.0],
                        "expected_shape": [2],
                    },
                    "reference": {
                        "source": "value",
                        "value": [1.0, 2.0],
                        "expected_shape": [2],
                    },
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="missing required helper input bindings"):
        evaluate_metric(tmp_path, contract, contract["metrics"][0], output_payload=None)
