from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
MYEVOSKILL_ROOT = ROOT / "MyEvoSkill"
ARTIFACTS_DIR = MYEVOSKILL_ROOT / "artifacts"
TASKS_DIR = ROOT / "tasks"
OUTPUT_DIR = ARTIFACTS_DIR / "visualizations" / "passed_and_one_metric_failed"
SUMMARY_PATH = ARTIFACTS_DIR / "task_total_table_57.json"


@dataclass(frozen=True)
class SelectionSpec:
    task_id: str
    category: str
    failed_metrics: list[str]
    run_id: str


@dataclass(frozen=True)
class ComparisonPayload:
    task_id: str
    kind: str
    gt: np.ndarray
    baseline: np.ndarray
    agent: np.ndarray
    display_name: str
    baseline_name: str
    agent_name: str
    baseline_note: str = ""
    agent_note: str = ""


def _load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _selected_tasks(summary: dict[str, Any]) -> list[SelectionSpec]:
    selected: list[SelectionSpec] = []
    for row in summary["rows"]:
        if row["category"] == "passed" or (
            row["category"] == "metric_failed" and len(row["failed_metrics"]) == 1
        ):
            selected.append(
                SelectionSpec(
                    task_id=row["task_id"],
                    category=row["category"],
                    failed_metrics=list(row["failed_metrics"]),
                    run_id=str(row["run_id"] or ""),
                )
            )
    return selected


def _workspace_output_path(task_id: str, run_id: str) -> Path:
    return MYEVOSKILL_ROOT / "artifacts" / "workspaces" / task_id / run_id / "output"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _squeeze_first_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _ensure_2d_image(array: np.ndarray, *, prefer_last: bool = False) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.real_if_close(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if prefer_last:
            index = arr.shape[-1] // 2
            return arr[..., index]
        index = arr.shape[0] // 2
        return arr[index]
    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size)))
        if side * side == arr.size:
            return arr.reshape(side, side)
    raise ValueError(f"cannot convert array with shape {list(arr.shape)} into a 2D image")


def _cube_to_rgb(array: np.ndarray) -> np.ndarray:
    cube = np.asarray(array, dtype=np.float64)
    while cube.ndim > 3 and cube.shape[0] == 1:
        cube = cube[0]
    if cube.ndim != 3:
        raise ValueError(f"expected 3D cube for RGB rendering, got shape {list(cube.shape)}")
    band_indices = [min(cube.shape[-1] - 1, index) for index in (24, cube.shape[-1] // 2, 6)]
    rgb = np.stack([cube[..., band_indices[0]], cube[..., band_indices[1]], cube[..., band_indices[2]]], axis=-1)
    rgb = rgb - np.nanmin(rgb)
    denom = np.nanmax(rgb)
    if denom > 0:
        rgb = rgb / denom
    return np.clip(rgb, 0.0, 1.0)


def _ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    denom = np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12
    return float(np.dot(est, ref) / denom)


def _nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    dynamic_range = float(ref.max() - ref.min() + 1e-12)
    return float(np.sqrt(np.mean((est - ref) ** 2)) / dynamic_range)


def _mae(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))


def _task_root(task_id: str) -> Path:
    return TASKS_DIR / task_id


def _reference_dir(task_id: str) -> Path:
    return _task_root(task_id) / "evaluation" / "reference_outputs"


def _data_dir(task_id: str) -> Path:
    return _task_root(task_id) / "data"


def _agent_output_npz(task_id: str, run_id: str, filename: str = "reconstruction.npz") -> dict[str, np.ndarray]:
    return _load_npz(_workspace_output_path(task_id, run_id) / filename)


def _load_mri_t2_mapping(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("mri_t2_mapping") / "ground_truth.npz")["T2_map"]
    baseline = _load_npz(_reference_dir("mri_t2_mapping") / "T2_map_loglinear.npz")["T2_map"]
    agent = _agent_output_npz("mri_t2_mapping", run_id)["T2_map"]
    return ComparisonPayload(
        task_id="mri_t2_mapping",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="T2 map",
        baseline_name="Log-linear baseline",
        agent_name="Agent reconstruction",
    )


def _load_mri_varnet(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("mri_varnet") / "ground_truth.npz")["image"]
    baseline = _load_npz(_reference_dir("mri_varnet") / "zerofill.npz")["reconstruction"]
    agent = _agent_output_npz("mri_varnet", run_id)["reconstruction"]
    return ComparisonPayload(
        task_id="mri_varnet",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Magnitude image",
        baseline_name="Zero-fill baseline",
        agent_name="Agent reconstruction",
    )


def _load_ct_sparse_view(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("ct_sparse_view") / "ground_truth.npz")["phantom"]
    baseline = _load_npz(_reference_dir("ct_sparse_view") / "reconstructions.npz")["fbp_sparse"]
    agent = _agent_output_npz("ct_sparse_view", run_id)["phantom"]
    return ComparisonPayload(
        task_id="ct_sparse_view",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="CT slice",
        baseline_name="Sparse-view FBP",
        agent_name="Agent reconstruction",
    )


def _load_diffusion_mri_dti(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("diffusion_mri_dti") / "ground_truth.npz")["fa_map"]
    baseline = _load_npz(_reference_dir("diffusion_mri_dti") / "dti_ols.npz")["fa_map"]
    agent = _agent_output_npz("diffusion_mri_dti", run_id)["fa_map"]
    return ComparisonPayload(
        task_id="diffusion_mri_dti",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="FA map",
        baseline_name="OLS baseline",
        agent_name="Agent reconstruction",
    )


def _load_eht_black_hole_feature_extraction_dynamic(run_id: str) -> ComparisonPayload:
    gt_data = _load_npz(_data_dir("eht_black_hole_feature_extraction_dynamic") / "ground_truth.npz")
    params = np.load(_reference_dir("eht_black_hole_feature_extraction_dynamic") / "all_params.npy", allow_pickle=True)
    weights = np.load(_reference_dir("eht_black_hole_feature_extraction_dynamic") / "all_weights.npy", allow_pickle=True)
    baseline = np.sum(params[:, :, 3] * weights, axis=1) / np.sum(weights, axis=1)
    agent = _agent_output_npz("eht_black_hole_feature_extraction_dynamic", run_id)["position_angle_deg"]
    return ComparisonPayload(
        task_id="eht_black_hole_feature_extraction_dynamic",
        kind="curve1d",
        gt=np.asarray(gt_data["position_angle_deg"], dtype=np.float64),
        baseline=np.asarray(baseline, dtype=np.float64),
        agent=np.asarray(agent, dtype=np.float64),
        display_name="Position angle trajectory",
        baseline_name="Reference posterior mean",
        agent_name="Agent prediction",
        baseline_note="Derived from weighted mean of reference posterior samples.",
    )


def _load_eht_black_hole_original(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("eht_black_hole_original") / "ground_truth.npz")["image"]
    metrics = _load_json(_reference_dir("eht_black_hole_original") / "metrics.json")
    corrupted_candidates = [
        ("closure-only_corrupt.npy", float(metrics["Closure-only (corrupt)"]["nrmse"])),
        ("amp_cp_corrupt.npy", float(metrics["Amp+CP (corrupt)"]["nrmse"])),
        ("vis_rml_corrupt.npy", float(metrics["Vis RML (corrupt)"]["nrmse"])),
    ]
    baseline_filename = min(corrupted_candidates, key=lambda item: item[1])[0]
    baseline = np.load(_reference_dir("eht_black_hole_original") / baseline_filename, allow_pickle=True)
    agent = _agent_output_npz("eht_black_hole_original", run_id)["image"]
    return ComparisonPayload(
        task_id="eht_black_hole_original",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Black hole image",
        baseline_name=f"Best corrupt reference ({baseline_filename.replace('.npy', '')})",
        agent_name="Agent reconstruction",
    )


def _load_mri_sense(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("mri_sense") / "ground_truth.npz")["image"]
    baseline = _load_npz(_reference_dir("mri_sense") / "zerofill.npz")["reconstruction"]
    agent = _agent_output_npz("mri_sense", run_id)["image"]
    return ComparisonPayload(
        task_id="mri_sense",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Magnitude image",
        baseline_name="Zero-fill baseline",
        agent_name="Agent reconstruction",
    )


def _load_pet_mlem(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("pet_mlem") / "ground_truth.npz")["activity_map"]
    baseline = _load_npz(_reference_dir("pet_mlem") / "recon_osem.npz")["reconstruction"]
    agent = _agent_output_npz("pet_mlem", run_id)["activity_map"]
    return ComparisonPayload(
        task_id="pet_mlem",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Activity map",
        baseline_name="OSEM baseline",
        agent_name="Agent MLEM",
        baseline_note="OSEM is used here as the available classical baseline for side-by-side comparison.",
    )


def _load_seismic_traveltime_tomography(run_id: str) -> ComparisonPayload:
    gt_velocity = _load_npz(_data_dir("seismic_traveltime_tomography") / "ground_truth.npz")["velocity"]
    meta = _load_json(_data_dir("seismic_traveltime_tomography") / "meta_data.json")
    nz = int(meta["Nz"])
    nx = int(meta["Nx"])
    v0 = float(meta["v0_km_s"])
    v1 = float(meta["v1_km_s"])
    background = np.linspace(v0, v1, nz, dtype=np.float64)[:, None] * np.ones((1, nx), dtype=np.float64)
    gt_perturbation = _ensure_2d_image(gt_velocity) - background
    baseline = _load_npz(_reference_dir("seismic_traveltime_tomography") / "baseline_reference.npz")[
        "velocity_perturbation"
    ]
    agent = _agent_output_npz("seismic_traveltime_tomography", run_id, filename="velocity_reconstructed.npz")[
        "velocity_perturbation"
    ]
    return ComparisonPayload(
        task_id="seismic_traveltime_tomography",
        kind="image2d",
        gt=np.asarray(gt_perturbation, dtype=np.float64),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Velocity perturbation",
        baseline_name="Baseline ATT",
        agent_name="Agent reconstruction",
        baseline_note="Ground truth perturbation is derived from ground-truth velocity minus the linear background model in meta_data.json.",
    )


def _load_shack_hartmann(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("shack-hartmann") / "ground_truth.npz")["wavefront_phases"]
    baseline = _load_npz(_reference_dir("shack-hartmann") / "reconstruction.npz")["reconstructed_phases"]
    agent = _agent_output_npz("shack-hartmann", run_id)["reconstructed_phases"]
    gt_level0 = _ensure_2d_image(_squeeze_first_axis(gt)[0])
    baseline_level0 = _ensure_2d_image(_squeeze_first_axis(baseline)[0])
    agent_level0 = _ensure_2d_image(_squeeze_first_axis(agent)[0])
    return ComparisonPayload(
        task_id="shack-hartmann",
        kind="image2d",
        gt=gt_level0,
        baseline=baseline_level0,
        agent=agent_level0,
        display_name="Wavefront phase (level 0)",
        baseline_name="Reference recon",
        agent_name="Agent reconstruction",
        baseline_note="Only level-0 phase map is shown here for readability.",
    )


def _load_shapelet_source_reconstruction(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("shapelet_source_reconstruction") / "ground_truth.npz")["source_image"]
    baseline = _load_npz(_reference_dir("shapelet_source_reconstruction") / "lensing_outputs.npz")["source_recon_2d"]
    agent = _agent_output_npz("shapelet_source_reconstruction", run_id)["source_image"]
    return ComparisonPayload(
        task_id="shapelet_source_reconstruction",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Source image",
        baseline_name="Reference source recon",
        agent_name="Agent reconstruction",
    )


def _load_spectral_snapshot_compressive_imaging(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("spectral_snapshot_compressive_imaging") / "ground_truth.npz")["hyperspectral_cube"]
    baseline = loadmat(_reference_dir("spectral_snapshot_compressive_imaging") / "kaist_crop256_01_result.mat")["img"]
    agent = _agent_output_npz("spectral_snapshot_compressive_imaging", run_id)["hyperspectral_cube"]
    return ComparisonPayload(
        task_id="spectral_snapshot_compressive_imaging",
        kind="rgb_cube",
        gt=np.asarray(_squeeze_first_axis(gt), dtype=np.float64),
        baseline=np.asarray(baseline, dtype=np.float64),
        agent=np.asarray(_squeeze_first_axis(agent), dtype=np.float64),
        display_name="Hyperspectral RGB preview",
        baseline_name="Reference recon",
        agent_name="Agent reconstruction",
        baseline_note="Visualization uses a pseudo-RGB rendering of the 31-band cube.",
    )


def _load_ultrasound_sos_tomography(run_id: str) -> ComparisonPayload:
    gt = _load_npz(_data_dir("ultrasound_sos_tomography") / "ground_truth.npz")["sos_phantom"]
    baseline = _load_npz(_reference_dir("ultrasound_sos_tomography") / "reconstructions.npz")["sos_fbp"]
    agent = _agent_output_npz("ultrasound_sos_tomography", run_id)["sos_phantom"]
    return ComparisonPayload(
        task_id="ultrasound_sos_tomography",
        kind="image2d",
        gt=_ensure_2d_image(gt),
        baseline=_ensure_2d_image(baseline),
        agent=_ensure_2d_image(agent),
        display_name="Speed-of-sound map",
        baseline_name="FBP baseline",
        agent_name="Agent reconstruction",
    )


LOADERS: dict[str, Callable[[str], ComparisonPayload]] = {
    "mri_t2_mapping": _load_mri_t2_mapping,
    "mri_varnet": _load_mri_varnet,
    "ct_sparse_view": _load_ct_sparse_view,
    "diffusion_mri_dti": _load_diffusion_mri_dti,
    "eht_black_hole_feature_extraction_dynamic": _load_eht_black_hole_feature_extraction_dynamic,
    "eht_black_hole_original": _load_eht_black_hole_original,
    "mri_sense": _load_mri_sense,
    "pet_mlem": _load_pet_mlem,
    "seismic_traveltime_tomography": _load_seismic_traveltime_tomography,
    "shack-hartmann": _load_shack_hartmann,
    "shapelet_source_reconstruction": _load_shapelet_source_reconstruction,
    "spectral_snapshot_compressive_imaging": _load_spectral_snapshot_compressive_imaging,
    "ultrasound_sos_tomography": _load_ultrasound_sos_tomography,
}


SKIP_REASONS = {
    "confocal-nlos-fk": "No ground truth is provided; only a baseline reference volume is available.",
    "xray_ptychography_tike": "No ground truth is provided; only a baseline reference reconstruction is available.",
}


def _render_image_comparison(payload: ComparisonPayload, destination: Path) -> dict[str, float]:
    gt = np.asarray(payload.gt, dtype=np.float64)
    baseline = np.asarray(payload.baseline, dtype=np.float64)
    agent = np.asarray(payload.agent, dtype=np.float64)

    baseline_metrics = {"ncc": _ncc(baseline, gt), "nrmse": _nrmse(baseline, gt)}
    agent_metrics = {"ncc": _ncc(agent, gt), "nrmse": _nrmse(agent, gt)}

    gt_display = gt
    baseline_display = baseline
    agent_display = agent
    if payload.kind == "rgb_cube":
        gt_display = _cube_to_rgb(gt)
        baseline_display = _cube_to_rgb(baseline)
        agent_display = _cube_to_rgb(agent)
        baseline_error = np.mean(np.abs(baseline - gt), axis=-1)
        agent_error = np.mean(np.abs(agent - gt), axis=-1)
    else:
        baseline_error = np.abs(baseline - gt)
        agent_error = np.abs(agent - gt)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        (axes[0, 0], gt_display, f"GT\n{payload.display_name}"),
        (
            axes[0, 1],
            baseline_display,
            f"{payload.baseline_name}\nNCC={baseline_metrics['ncc']:.4f}, NRMSE={baseline_metrics['nrmse']:.4f}",
        ),
        (
            axes[0, 2],
            agent_display,
            f"{payload.agent_name}\nNCC={agent_metrics['ncc']:.4f}, NRMSE={agent_metrics['nrmse']:.4f}",
        ),
        (axes[1, 0], np.zeros_like(baseline_error), " "),
        (axes[1, 1], baseline_error, "|Baseline - GT|"),
        (axes[1, 2], agent_error, "|Agent - GT|"),
    ]
    for ax, image, title in panels:
        ax.axis("off")
        if title.strip():
            ax.set_title(title, fontsize=10)
        if image.ndim == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")
    notes = [note for note in (payload.baseline_note, payload.agent_note) if note]
    fig.suptitle(payload.task_id, fontsize=14)
    if notes:
        fig.text(0.5, 0.02, " | ".join(notes), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if notes else 0, 1, 0.96))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_ncc": baseline_metrics["ncc"],
        "baseline_nrmse": baseline_metrics["nrmse"],
        "agent_ncc": agent_metrics["ncc"],
        "agent_nrmse": agent_metrics["nrmse"],
    }


def _render_curve_comparison(payload: ComparisonPayload, destination: Path) -> dict[str, float]:
    gt = np.asarray(payload.gt, dtype=np.float64).reshape(-1)
    baseline = np.asarray(payload.baseline, dtype=np.float64).reshape(-1)
    agent = np.asarray(payload.agent, dtype=np.float64).reshape(-1)
    x = np.arange(gt.size)
    baseline_mae = _mae(baseline, gt)
    agent_mae = _mae(agent, gt)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, gt, marker="o", linewidth=2, label="GT")
    ax.plot(x, baseline, marker="s", linewidth=2, label=f"{payload.baseline_name} (MAE={baseline_mae:.3f})")
    ax.plot(x, agent, marker="^", linewidth=2, label=f"{payload.agent_name} (MAE={agent_mae:.3f})")
    ax.set_title(f"{payload.task_id}: {payload.display_name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Degrees")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    notes = [note for note in (payload.baseline_note, payload.agent_note) if note]
    if notes:
        fig.text(0.5, 0.01, " | ".join(notes), ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_mae": baseline_mae,
        "agent_mae": agent_mae,
    }


def _render_one(spec: SelectionSpec) -> dict[str, Any]:
    if spec.task_id in SKIP_REASONS:
        return {
            "task_id": spec.task_id,
            "status": "skipped",
            "reason": SKIP_REASONS[spec.task_id],
            "category": spec.category,
            "failed_metrics": spec.failed_metrics,
        }
    if spec.task_id not in LOADERS:
        return {
            "task_id": spec.task_id,
            "status": "skipped",
            "reason": "No visualization loader has been defined for this task.",
            "category": spec.category,
            "failed_metrics": spec.failed_metrics,
        }

    payload = LOADERS[spec.task_id](spec.run_id)
    destination = OUTPUT_DIR / f"{spec.task_id}.png"
    if payload.kind == "curve1d":
        metrics = _render_curve_comparison(payload, destination)
    else:
        metrics = _render_image_comparison(payload, destination)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "category": spec.category,
        "failed_metrics": spec.failed_metrics,
        "figure_path": str(destination),
        "metrics": metrics,
        "baseline_note": payload.baseline_note,
        "agent_note": payload.agent_note,
    }


def _write_report(records: list[dict[str, Any]]) -> None:
    report_path = OUTPUT_DIR / "README.md"
    lines = [
        "# Passed + One-Metric-Failed Visual Comparison",
        "",
        "This report compares ground truth, a task-specific baseline, and the latest live-run agent output.",
        "",
    ]
    for record in records:
        task_id = record["task_id"]
        lines.append(f"## {task_id}")
        lines.append("")
        lines.append(f"- Status: `{record['status']}`")
        lines.append(f"- Category: `{record['category']}`")
        lines.append(f"- Failed metrics: `{record.get('failed_metrics', [])}`")
        if record["status"] == "rendered":
            figure_name = Path(record["figure_path"]).name
            lines.append(f"- Figure: ![{task_id}]({figure_name})")
            if record.get("metrics"):
                lines.append(f"- Summary metrics: `{json.dumps(record['metrics'], ensure_ascii=False)}`")
            for note_key in ("baseline_note", "agent_note"):
                note = record.get(note_key) or ""
                if note:
                    lines.append(f"- Note: {note}")
        else:
            lines.append(f"- Reason: {record['reason']}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    selected = _selected_tasks(summary)
    records = [_render_one(spec) for spec in selected]
    report = {
        "selected_count": len(selected),
        "rendered_count": sum(1 for item in records if item["status"] == "rendered"),
        "skipped_count": sum(1 for item in records if item["status"] == "skipped"),
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
