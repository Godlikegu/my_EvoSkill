"""Notebook-style workspace visualization for eht_black_hole_feature_extraction_dynamic.

This script scans `MyEvoSkill/artifacts/workspaces/eht_black_hole_feature_extraction_dynamic`,
loads each run's predicted `position_angle_deg`, regenerates the corresponding crescent
images using the task's ground-truth geometric parameters, and saves a notebook-style
two-row montage:

1. Predicted images over time
2. Ground-truth images over time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MYEVOSKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MYEVOSKILL_ROOT.parent
TASK_ID = "eht_black_hole_feature_extraction_dynamic"
TASK_ROOT = REPO_ROOT / "tasks" / TASK_ID
WORKSPACE_ROOT = MYEVOSKILL_ROOT / "artifacts" / "workspaces" / TASK_ID
OUTPUT_ROOT = MYEVOSKILL_ROOT / "artifacts" / "visualizations" / "workspaces" / TASK_ID

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        help="Specific workspace run to render, e.g. run-live-1776773393. May be provided multiple times.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Render only the latest available workspace run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Directory for PNG outputs and summary.json. Default: {OUTPUT_ROOT}",
    )
    return parser.parse_args()


def generate_simple_crescent_image(
    npix: int,
    fov_uas: float,
    diameter_uas: float,
    width_uas: float,
    asymmetry: float,
    pa_deg: float,
) -> np.ndarray:
    """Rebuild the task's simple crescent image without importing optional task dependencies."""
    half_fov = 0.5 * fov_uas
    eps = 1e-4

    gap = 1.0 / npix
    xs = np.arange(-1 + gap, 1, 2 * gap)
    grid_y, grid_x = np.meshgrid(-xs, xs, indexing="ij")
    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    grid_theta = np.arctan2(grid_y, grid_x)

    r = (diameter_uas / 2.0) / half_fov
    sigma = width_uas / half_fov
    eta = pa_deg * np.pi / 180.0

    ring = np.exp(-0.5 * (grid_r - r) ** 2 / sigma**2)
    brightness = 1.0 + asymmetry * np.cos(grid_theta - eta)
    image = brightness * ring
    return image / (image.sum() + eps)


def _load_ground_truth() -> dict[str, np.ndarray | float]:
    metadata = json.loads((TASK_ROOT / "data" / "meta_data.json").read_text(encoding="utf-8"))
    gt_npz = np.load(TASK_ROOT / "data" / "ground_truth.npz", allow_pickle=True)
    return {
        "frame_times_hr": np.asarray(metadata["frame_times_hr"], dtype=np.float64),
        "npix": int(metadata["npix"]),
        "fov_uas": float(metadata["fov_uas"]),
        "diameter_uas": np.asarray(gt_npz["diameter_uas"], dtype=np.float64),
        "width_uas": np.asarray(gt_npz["width_uas"], dtype=np.float64),
        "asymmetry": np.asarray(gt_npz["asymmetry"], dtype=np.float64),
        "position_angle_deg": np.asarray(gt_npz["position_angle_deg"], dtype=np.float64),
        "images": np.asarray(gt_npz["images"], dtype=np.float64),
    }


def _list_runs(workspace_root: Path, run_ids: list[str] | None, latest_only: bool) -> list[Path]:
    if run_ids:
        runs = [workspace_root / run_id for run_id in run_ids]
    else:
        runs = [path for path in workspace_root.iterdir() if path.is_dir() and path.name.startswith("run-live-")]
        runs.sort(key=lambda path: path.name)
        if latest_only and runs:
            runs = [runs[-1]]

    missing = [str(path) for path in runs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Workspace run(s) not found: {missing}")
    return runs


def _load_prediction(run_dir: Path, expected_frames: int) -> np.ndarray:
    prediction_path = run_dir / "output" / "reconstruction.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing reconstruction output: {prediction_path}")

    prediction = np.asarray(np.load(prediction_path, allow_pickle=True)["position_angle_deg"], dtype=np.float64)
    if prediction.shape != (expected_frames,):
        raise ValueError(
            f"{prediction_path} has shape {prediction.shape}, expected {(expected_frames,)} for position_angle_deg."
        )
    return prediction


def _wrap_angle_diff_deg(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    return (prediction - ground_truth + 180.0) % 360.0 - 180.0


def _reconstruct_images(prediction_deg: np.ndarray, gt: dict[str, np.ndarray | float]) -> np.ndarray:
    npix = int(gt["npix"])
    fov_uas = float(gt["fov_uas"])
    diameter = np.asarray(gt["diameter_uas"], dtype=np.float64)
    width = np.asarray(gt["width_uas"], dtype=np.float64)
    asymmetry = np.asarray(gt["asymmetry"], dtype=np.float64)

    return np.stack(
        [
            generate_simple_crescent_image(
                npix=npix,
                fov_uas=fov_uas,
                diameter_uas=float(diameter[i]),
                width_uas=float(width[i]),
                asymmetry=float(asymmetry[i]),
                pa_deg=float(prediction_deg[i]),
            )
            for i in range(len(prediction_deg))
        ],
        axis=0,
    )


def _plot_notebook_style(
    prediction_images: np.ndarray,
    gt_images: np.ndarray,
    frame_times_hr: np.ndarray,
    fov_uas: float,
    output_path: Path,
    run_id: str,
    angle_mae_deg: float,
) -> None:
    n_frames = prediction_images.shape[0]
    n_show = min(n_frames, 10)
    frame_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    half = fov_uas / 2.0
    extent = [half, -half, -half, half]

    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 4.8))
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(frame_indices):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        ax_top.imshow(prediction_images[idx], origin="lower", cmap="afmhot", extent=extent)
        ax_top.set_title(f"t={frame_times_hr[idx]:.1f}h", fontsize=9)
        ax_top.tick_params(labelsize=7, length=2)
        if col == 0:
            ax_top.set_ylabel("Prediction", fontsize=9)

        ax_bottom.imshow(gt_images[idx], origin="lower", cmap="afmhot", extent=extent)
        ax_bottom.tick_params(labelsize=7, length=2)
        if col == 0:
            ax_bottom.set_ylabel("Ground truth", fontsize=9)

    fig.suptitle(f"{TASK_ID}\n{run_id} | PA MAE = {angle_mae_deg:.2f} deg", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    gt = _load_ground_truth()
    runs = _list_runs(WORKSPACE_ROOT, args.run_ids, args.latest)
    if not runs:
        raise FileNotFoundError(f"No workspace runs found under {WORKSPACE_ROOT}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_angles = np.asarray(gt["position_angle_deg"], dtype=np.float64)
    gt_images = np.asarray(gt["images"], dtype=np.float64)
    frame_times_hr = np.asarray(gt["frame_times_hr"], dtype=np.float64)
    fov_uas = float(gt["fov_uas"])

    rendered: list[dict[str, object]] = []
    for run_dir in runs:
        prediction = _load_prediction(run_dir, expected_frames=len(gt_angles))
        angle_diff = _wrap_angle_diff_deg(prediction, gt_angles)
        prediction_images = _reconstruct_images(prediction, gt)
        output_path = output_dir / f"{run_dir.name}.png"

        _plot_notebook_style(
            prediction_images=prediction_images,
            gt_images=gt_images,
            frame_times_hr=frame_times_hr,
            fov_uas=fov_uas,
            output_path=output_path,
            run_id=run_dir.name,
            angle_mae_deg=float(np.mean(np.abs(angle_diff))),
        )

        rendered.append(
            {
                "run_id": run_dir.name,
                "figure_path": str(output_path),
                "n_frames": int(len(gt_angles)),
                "position_angle_mae_deg": float(np.mean(np.abs(angle_diff))),
                "position_angle_max_abs_err_deg": float(np.max(np.abs(angle_diff))),
                "prediction_start_deg": float(prediction[0]),
                "prediction_end_deg": float(prediction[-1]),
                "ground_truth_start_deg": float(gt_angles[0]),
                "ground_truth_end_deg": float(gt_angles[-1]),
                "note": (
                    "Images are regenerated from workspace-predicted position_angle_deg together "
                    "with ground-truth diameter_uas, width_uas, and asymmetry, because workspace "
                    "outputs do not contain all_images.npy."
                ),
            }
        )

    summary = {
        "task_id": TASK_ID,
        "workspace_root": str(WORKSPACE_ROOT),
        "output_dir": str(output_dir),
        "rendered_runs": rendered,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rendered {len(rendered)} run(s) to {output_dir}")
    for item in rendered:
        print(
            f"  {item['run_id']}: {item['figure_path']} "
            f"(PA MAE={item['position_angle_mae_deg']:.2f} deg)"
        )


if __name__ == "__main__":
    main()
