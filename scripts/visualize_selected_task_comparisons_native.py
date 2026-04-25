from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import convolve


ROOT = Path(__file__).resolve().parents[2]
MYEVOSKILL_ROOT = ROOT / "MyEvoSkill"
TASKS_DIR = ROOT / "tasks"
ARTIFACTS_DIR = MYEVOSKILL_ROOT / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "task_total_table_57.json"
OUTPUT_DIR = ARTIFACTS_DIR / "visualizations" / "passed_and_one_metric_failed"
PARTS_DIR = OUTPUT_DIR / "_native_parts"


@dataclass(frozen=True)
class SelectionSpec:
    task_id: str
    category: str
    failed_metrics: list[str]
    run_id: str


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


def _task_root(task_id: str) -> Path:
    return TASKS_DIR / task_id


def _task_module(task_id: str):
    module_path = _task_root(task_id) / "src" / "visualization.py"
    spec = importlib.util.spec_from_file_location(f"task_vis_{task_id.replace('-', '_')}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import visualization module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    if task_id == "spectral_snapshot_compressive_imaging" and "cv2" not in sys.modules:
        cv2_stub = types.SimpleNamespace()

        def _get_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
            center = (ksize - 1) / 2.0
            x = np.arange(ksize, dtype=np.float64) - center
            kernel = np.exp(-(x**2) / (2.0 * sigma**2))
            kernel = kernel / np.sum(kernel)
            return kernel.reshape(-1, 1)

        def _filter_2d(image: np.ndarray, ddepth: int, kernel: np.ndarray) -> np.ndarray:
            del ddepth
            return convolve(np.asarray(image, dtype=np.float64), np.asarray(kernel, dtype=np.float64), mode="reflect")

        cv2_stub.getGaussianKernel = _get_gaussian_kernel
        cv2_stub.filter2D = _filter_2d
        sys.modules["cv2"] = cv2_stub
    spec.loader.exec_module(module)
    return module


def _task_relative_module(task_id: str, relative_path: str, module_name: str):
    module_path = _task_root(task_id) / relative_path
    spec = importlib.util.spec_from_file_location(
        f"task_{module_name}_{task_id.replace('-', '_')}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_output_path(task_id: str, run_id: str, filename: str = "reconstruction.npz") -> Path:
    return MYEVOSKILL_ROOT / "artifacts" / "workspaces" / task_id / run_id / "output" / filename


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _squeeze_first_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _ensure_leading_slice_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def _save_native_figure(fig: Any, destination: Path) -> None:
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _concat_images(image_paths: list[Path], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_height = max(image.height for image in images)
    widths = []
    padded_images: list[Image.Image] = []
    for image in images:
        if image.height != max_height:
            canvas = Image.new("RGB", (image.width, max_height), "white")
            canvas.paste(image, (0, (max_height - image.height) // 2))
            image = canvas
        padded_images.append(image)
        widths.append(image.width)
    stitched = Image.new("RGB", (sum(widths), max_height), "white")
    offset = 0
    for image in padded_images:
        stitched.paste(image, (offset, 0))
        offset += image.width
    stitched.save(destination)


def _ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    denom = np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12
    return float(np.dot(est, ref) / denom)


def _nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean((est - ref) ** 2)) / (ref.max() - ref.min() + 1e-12))


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    mse = float(np.mean((ref - est) ** 2))
    if mse <= 1e-30:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _ssim_mean(module: Any, reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(module.calculate_ssim(np.asarray(reference), np.asarray(estimate)))


def _render_mri_t2_mapping(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt_data = _load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")
    baseline_data = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "T2_map_loglinear.npz")
    agent_data = _load_npz(_workspace_output_path(spec.task_id, spec.run_id))
    gt = _squeeze_first_axis(gt_data["T2_map"])
    mask = _squeeze_first_axis(gt_data["tissue_mask"]).astype(bool)
    baseline = _squeeze_first_axis(baseline_data["T2_map"])
    agent = _squeeze_first_axis(agent_data["T2_map"])

    baseline_path = PARTS_DIR / f"{spec.task_id}_baseline.png"
    agent_path = PARTS_DIR / f"{spec.task_id}_agent.png"
    module.plot_t2_maps(gt, baseline, mask, title_est="Baseline T2", save_path=str(baseline_path))
    module.plot_t2_maps(gt, agent, mask, title_est="Agent T2", save_path=str(agent_path))

    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _concat_images([baseline_path, agent_path], output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc": module.compute_ncc(baseline, gt, mask),
            "baseline_nrmse": module.compute_nrmse(baseline, gt, mask),
            "agent_ncc": module.compute_ncc(agent, gt, mask),
            "agent_nrmse": module.compute_nrmse(agent, gt, mask),
        },
    }


def _render_mri_varnet(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _ensure_leading_slice_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["image"])
    baseline = _ensure_leading_slice_axis(_load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "zerofill.npz")["reconstruction"])
    agent = _ensure_leading_slice_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["reconstruction"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_reconstruction_comparison(agent, baseline, gt, slice_idx=0, save_path=str(output_path))
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline": module.compute_metrics(_squeeze_first_axis(baseline), _squeeze_first_axis(gt)),
            "agent": module.compute_metrics(_squeeze_first_axis(agent), _squeeze_first_axis(gt)),
        },
    }


def _render_confocal_nlos_fk(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    pre = _task_relative_module(spec.task_id, "src/preprocessing.py", "preprocessing")
    raw = _load_npz(_task_root(spec.task_id) / "data" / "raw_data.npz")
    meta = _load_json(_task_root(spec.task_id) / "data" / "meta_data")
    baseline = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "baseline_reference.npz")["reconstruction"])
    agent = _load_npz(_workspace_output_path(spec.task_id, spec.run_id))["fk"]
    wall_size = float(np.asarray(raw["wall_size"]).reshape(-1)[0])
    bin_resolution = float(np.asarray(raw["bin_resolution"]).reshape(-1)[0])
    meas = pre.preprocess_measurements(
        np.asarray(raw["meas"]),
        np.asarray(raw["tofgrid"]),
        bin_resolution=bin_resolution,
        crop=int(meta.get("n_time_crop", 512)),
    )
    fig = module.plot_notebook_comparison(
        meas=np.asarray(meas),
        baseline_vol=np.asarray(baseline),
        agent_vol=np.asarray(agent),
        wall_size=wall_size,
        bin_resolution=bin_resolution,
    )
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _save_native_figure(fig, output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_vs_agent_ncc": _ncc(agent, baseline),
            "baseline_vs_agent_nrmse": _nrmse(agent, baseline),
        },
        "note": "No ground truth exists for this task; notebook-style comparison uses the preprocessed transient slice plus baseline and agent f-k volumes.",
    }


def _render_ct_sparse_view(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["phantom"])
    ref_npz = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "reconstructions.npz")
    baseline = _squeeze_first_axis(ref_npz["fbp_sparse"])
    agent = _squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["phantom"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_reconstruction_comparison(gt, baseline, agent, save_path=str(output_path))
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc": module.compute_ncc(baseline, gt),
            "baseline_nrmse": module.compute_nrmse(baseline, gt),
            "agent_ncc": module.compute_ncc(agent, gt),
            "agent_nrmse": module.compute_nrmse(agent, gt),
        },
    }


def _render_diffusion_mri_dti(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt_npz = _load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")
    baseline_npz = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "dti_ols.npz")
    agent_npz = _load_npz(_workspace_output_path(spec.task_id, spec.run_id))
    mask = _squeeze_first_axis(gt_npz["tissue_mask"]).astype(bool)
    fa_gt = _squeeze_first_axis(gt_npz["fa_map"])
    md_gt = _squeeze_first_axis(gt_npz["md_map"])
    fa_baseline = _squeeze_first_axis(baseline_npz["fa_map"])
    md_baseline = _squeeze_first_axis(baseline_npz["md_map"])
    fa_agent = _squeeze_first_axis(agent_npz["fa_map"])
    md_agent = _squeeze_first_axis(agent_npz["md_map"])

    baseline_path = PARTS_DIR / f"{spec.task_id}_baseline.png"
    agent_path = PARTS_DIR / f"{spec.task_id}_agent.png"
    module.plot_dti_maps(fa_gt, fa_baseline, md_gt, md_baseline, mask, title_est="OLS baseline", save_path=str(baseline_path))
    module.plot_dti_maps(fa_gt, fa_agent, md_gt, md_agent, mask, title_est="Agent reconstruction", save_path=str(agent_path))

    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _concat_images([baseline_path, agent_path], output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_fa_ncc": module.compute_ncc(fa_baseline, fa_gt, mask),
            "baseline_fa_nrmse": module.compute_nrmse(fa_baseline, fa_gt, mask),
            "agent_fa_ncc": module.compute_ncc(fa_agent, fa_gt, mask),
            "agent_fa_nrmse": module.compute_nrmse(fa_agent, fa_gt, mask),
        },
    }


def _render_eht_black_hole_feature_extraction_dynamic(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt_npz = _load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")
    ref_dir = _task_root(spec.task_id) / "evaluation" / "reference_outputs"
    params = np.load(ref_dir / "all_params.npy", allow_pickle=True)
    weights = np.load(ref_dir / "all_weights.npy", allow_pickle=True)
    frame_times = _load_json(_task_root(spec.task_id) / "data" / "meta_data.json")["frame_times_hr"]
    gt_per_frame = [
        {
            "diameter_uas": float(gt_npz["diameter_uas"][i]),
            "width_uas": float(gt_npz["width_uas"][i]),
            "asymmetry": float(gt_npz["asymmetry"][i]),
            "position_angle_deg": float(gt_npz["position_angle_deg"][i]),
        }
        for i in range(len(gt_npz["position_angle_deg"]))
    ]
    baseline_params = [np.asarray(params[i], dtype=np.float64) for i in range(params.shape[0])]
    baseline_weights = [np.asarray(weights[i], dtype=np.float64) for i in range(weights.shape[0])]
    agent_angles = np.asarray(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["position_angle_deg"], dtype=np.float64)
    agent_params = [
        np.asarray(
            [[
                gt_per_frame[i]["diameter_uas"],
                gt_per_frame[i]["width_uas"],
                gt_per_frame[i]["asymmetry"],
                float(agent_angles[i]),
            ]],
            dtype=np.float64,
        )
        for i in range(len(agent_angles))
    ]
    agent_weights = [np.asarray([1.0], dtype=np.float64) for _ in range(len(agent_angles))]

    baseline_path = PARTS_DIR / f"{spec.task_id}_baseline.png"
    agent_path = PARTS_DIR / f"{spec.task_id}_agent.png"
    module.plot_param_evolution(
        baseline_params,
        ["diameter_uas", "width_uas", "asymmetry", "position_angle_deg"],
        gt_per_frame,
        baseline_weights,
        frame_times,
        save_path=str(baseline_path),
    )
    module.plot_param_evolution(
        agent_params,
        ["diameter_uas", "width_uas", "asymmetry", "position_angle_deg"],
        gt_per_frame,
        agent_weights,
        frame_times,
        save_path=str(agent_path),
    )

    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _concat_images([baseline_path, agent_path], output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_position_angle_mae_deg": float(
                np.mean(np.abs(np.sum(params[:, :, 3] * weights, axis=1) / np.sum(weights, axis=1) - gt_npz["position_angle_deg"]))
            ),
            "agent_position_angle_mae_deg": float(np.mean(np.abs(agent_angles - gt_npz["position_angle_deg"]))),
        },
        "note": "Baseline figure uses native posterior evolution; agent figure uses a degenerate posterior centered at the agent prediction.",
    }


def _render_eht_black_hole_original(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = np.asarray(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["image"], dtype=np.float64)
    ref_dir = _task_root(spec.task_id) / "evaluation" / "reference_outputs"
    metrics_json = _load_json(ref_dir / "metrics.json")
    corrupted_candidates = [
        ("closure-only_corrupt.npy", float(metrics_json["Closure-only (corrupt)"]["nrmse"])),
        ("amp_cp_corrupt.npy", float(metrics_json["Amp+CP (corrupt)"]["nrmse"])),
        ("vis_rml_corrupt.npy", float(metrics_json["Vis RML (corrupt)"]["nrmse"])),
    ]
    baseline_name = min(corrupted_candidates, key=lambda item: item[1])[0]
    baseline = np.load(ref_dir / baseline_name, allow_pickle=True)
    agent = np.asarray(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["image"], dtype=np.float64)
    metrics_dict = {
        f"Baseline ({baseline_name.replace('.npy', '')})": module.compute_metrics(baseline, gt),
        "Agent reconstruction": module.compute_metrics(agent, gt),
    }
    fig = module.plot_comparison(
        images={
            f"Baseline ({baseline_name.replace('.npy', '')})": baseline,
            "Agent reconstruction": agent,
        },
        ground_truth=gt,
        metrics_dict=metrics_dict,
        suptitle="Reconstruction Comparison",
    )
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _save_native_figure(fig, output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": metrics_dict,
    }


def _render_mri_sense(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["image"])
    baseline = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "zerofill.npz")["reconstruction"])
    agent = _squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["image"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_reconstruction_comparison(agent, baseline, gt, save_path=str(output_path))
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline": module.compute_metrics(baseline, gt),
            "agent": module.compute_metrics(agent, gt),
        },
    }


def _render_pet_mlem(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["activity_map"])
    baseline = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "recon_osem.npz")["reconstruction"])
    agent = _squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["activity_map"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_pet_reconstruction(gt, agent, baseline, save_path=str(output_path))
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc": module.compute_ncc(baseline, gt),
            "baseline_nrmse": module.compute_nrmse(baseline, gt),
            "agent_ncc": module.compute_ncc(agent, gt),
            "agent_nrmse": module.compute_nrmse(agent, gt),
        },
        "note": "The native PET visualization labels the second and third panels as MLEM and OSEM; here they correspond to agent and baseline respectively.",
    }


def _render_seismic_traveltime_tomography(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    meta = _load_json(_task_root(spec.task_id) / "data" / "meta_data.json")
    raw = _load_npz(_task_root(spec.task_id) / "data" / "raw_data.npz")
    gt_velocity = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["velocity"])
    baseline_npz = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "baseline_reference.npz")
    agent_npz = _load_npz(_workspace_output_path(spec.task_id, spec.run_id, filename="velocity_reconstructed.npz"))

    nz = int(meta["Nz"])
    nx = int(meta["Nx"])
    background = np.linspace(float(meta["v0_km_s"]), float(meta["v1_km_s"]), nz, dtype=np.float64)[:, None] * np.ones((1, nx))
    baseline_velocity = _squeeze_first_axis(baseline_npz["velocity"])
    agent_velocity = background + _squeeze_first_axis(agent_npz["velocity_perturbation"])

    baseline_metrics = module.evaluate_reconstruction(baseline_velocity, gt_velocity, background)
    agent_metrics = module.evaluate_reconstruction(agent_velocity, gt_velocity, background)
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_notebook_comparison(
        gt_velocity,
        background,
        baseline_velocity,
        agent_velocity,
        float(meta["dx_km"]),
        float(meta["dz_km"]),
        sources=_squeeze_first_axis(raw["sources"]),
        receivers=_squeeze_first_axis(raw["receivers"]),
        baseline_metrics=baseline_metrics,
        agent_metrics=agent_metrics,
        save_path=str(output_path),
    )
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline": baseline_metrics,
            "agent": agent_metrics,
        },
    }


def _render_shack_hartmann(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    raw = _load_npz(_task_root(spec.task_id) / "data" / "raw_data.npz")
    gt_npz = _load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")
    baseline_npz = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "reconstruction.npz")
    agent_npz = _load_npz(_workspace_output_path(spec.task_id, spec.run_id))
    meta = _load_json(_task_root(spec.task_id) / "data" / "meta_data.json")

    gt = _squeeze_first_axis(gt_npz["wavefront_phases"])
    aperture = _squeeze_first_axis(raw["aperture"])
    pupil_shape = tuple(meta["wavefront_sensor"]["det_image_shape"])
    wfe_levels = list(meta["wfe_levels_nm"])
    baseline_recon = _squeeze_first_axis(baseline_npz["reconstructed_phases"])
    agent_recon = _squeeze_first_axis(agent_npz["reconstructed_phases"])

    baseline_path = PARTS_DIR / f"{spec.task_id}_baseline.png"
    agent_path = PARTS_DIR / f"{spec.task_id}_agent.png"
    module.plot_wavefront_comparison(
        gt, baseline_recon, aperture, pupil_shape, wfe_levels,
        ncc_arr=np.asarray(baseline_npz["ncc_per_level"]),
        nrmse_arr=np.asarray(baseline_npz["nrmse_per_level"]),
        output_path=str(baseline_path),
    )
    module.plot_wavefront_comparison(
        gt, agent_recon, aperture, pupil_shape, wfe_levels,
        output_path=str(agent_path),
    )
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _concat_images([baseline_path, agent_path], output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc_per_level": np.asarray(baseline_npz["ncc_per_level"]).tolist(),
            "baseline_nrmse_per_level": np.asarray(baseline_npz["nrmse_per_level"]).tolist(),
        },
    }


def _render_shapelet_source_reconstruction(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["source_image"])
    lens_outputs = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "lensing_outputs.npz")
    baseline = lens_outputs["source_recon_2d"]
    agent = _squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["source_image"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_notebook_comparison(
        lens_outputs=lens_outputs,
        agent_source=agent,
        baseline_source=baseline,
        ground_truth_source=gt,
        save_path=str(output_path),
    )
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc": _ncc(baseline, gt),
            "baseline_nrmse": _nrmse(baseline, gt),
            "agent_ncc": _ncc(agent, gt),
            "agent_nrmse": _nrmse(agent, gt),
        },
    }


def _render_spectral_snapshot_compressive_imaging(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = np.asarray(_squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["hyperspectral_cube"]), dtype=np.float64)
    baseline = np.asarray(loadmat(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "kaist_crop256_01_result.mat")["img"], dtype=np.float64)
    agent = np.asarray(_squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["hyperspectral_cube"]), dtype=np.float64)
    gt_norm = gt / max(float(gt.max()), 1e-12)
    baseline_norm = baseline / max(float(baseline.max()), 1e-12)
    agent_norm = agent / max(float(agent.max()), 1e-12)

    baseline_path = PARTS_DIR / f"{spec.task_id}_baseline.png"
    agent_path = PARTS_DIR / f"{spec.task_id}_agent.png"
    module.plot_comparison(gt_norm, baseline_norm, _psnr(gt_norm, baseline_norm), _ssim_mean(module, gt_norm, baseline_norm), save_path=str(baseline_path))
    module.plot_comparison(gt_norm, agent_norm, _psnr(gt_norm, agent_norm), _ssim_mean(module, gt_norm, agent_norm), save_path=str(agent_path))

    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    _concat_images([baseline_path, agent_path], output_path)
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_psnr": _psnr(gt_norm, baseline_norm),
            "baseline_ssim": _ssim_mean(module, gt_norm, baseline_norm),
            "agent_psnr": _psnr(gt_norm, agent_norm),
            "agent_ssim": _ssim_mean(module, gt_norm, agent_norm),
        },
    }


def _render_ultrasound_sos_tomography(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    gt = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "ground_truth.npz")["sos_phantom"])
    ref_npz = _load_npz(_task_root(spec.task_id) / "evaluation" / "reference_outputs" / "reconstructions.npz")
    baseline = _squeeze_first_axis(ref_npz["sos_fbp"])
    agent = _squeeze_first_axis(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["sos_phantom"])
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_sos_comparison(gt, baseline, agent, save_path=str(output_path))
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": {
            "baseline_ncc": module.compute_ncc(baseline, gt),
            "baseline_nrmse": module.compute_nrmse(baseline, gt),
            "agent_ncc": module.compute_ncc(agent, gt),
            "agent_nrmse": module.compute_nrmse(agent, gt),
        },
    }


def _render_xray_ptychography_tike(spec: SelectionSpec) -> dict[str, Any]:
    module = _task_module(spec.task_id)
    raw = _load_npz(_task_root(spec.task_id) / "data" / "raw_data.npz")
    baseline_phase = _squeeze_first_axis(_load_npz(_task_root(spec.task_id) / "data" / "baseline_reference.npz")["object_phase"])
    agent_phase = np.asarray(_load_npz(_workspace_output_path(spec.task_id, spec.run_id))["object_phase"], dtype=np.float64)
    metrics = module.compute_metrics(agent_phase, baseline_phase)
    output_path = OUTPUT_DIR / f"{spec.task_id}.png"
    module.plot_notebook_summary(
        scan=raw["scan_positions"],
        diffraction_patterns=raw["diffraction_patterns"],
        probe_guess=raw["probe_guess"],
        reference_phase=baseline_phase,
        estimate_phase=agent_phase,
        metrics=metrics,
        save_path=str(output_path),
    )
    return {
        "task_id": spec.task_id,
        "status": "rendered",
        "figure_path": str(output_path),
        "metrics": metrics,
        "note": "No ground truth exists for this task; notebook-style summary uses raw scan/diffraction inputs and baseline-vs-agent phase comparison.",
    }


RENDERERS: dict[str, Callable[[SelectionSpec], dict[str, Any]]] = {
    "mri_t2_mapping": _render_mri_t2_mapping,
    "mri_varnet": _render_mri_varnet,
    "confocal-nlos-fk": _render_confocal_nlos_fk,
    "ct_sparse_view": _render_ct_sparse_view,
    "diffusion_mri_dti": _render_diffusion_mri_dti,
    "eht_black_hole_feature_extraction_dynamic": _render_eht_black_hole_feature_extraction_dynamic,
    "eht_black_hole_original": _render_eht_black_hole_original,
    "mri_sense": _render_mri_sense,
    "pet_mlem": _render_pet_mlem,
    "seismic_traveltime_tomography": _render_seismic_traveltime_tomography,
    "shack-hartmann": _render_shack_hartmann,
    "shapelet_source_reconstruction": _render_shapelet_source_reconstruction,
    "spectral_snapshot_compressive_imaging": _render_spectral_snapshot_compressive_imaging,
    "ultrasound_sos_tomography": _render_ultrasound_sos_tomography,
    "xray_ptychography_tike": _render_xray_ptychography_tike,
}


def _write_report(records: list[dict[str, Any]]) -> None:
    lines = [
        "# Native Visualization Report",
        "",
        "Each figure below is generated by calling the task's own `src/visualization.py` functions.",
        "",
    ]
    for record in records:
        lines.append(f"## {record['task_id']}")
        lines.append("")
        lines.append(f"- Figure: ![{record['task_id']}]({Path(record['figure_path']).name})")
        if record.get("metrics") is not None:
            lines.append(f"- Metrics: `{json.dumps(record['metrics'], ensure_ascii=False)}`")
        if record.get("note"):
            lines.append(f"- Note: {record['note']}")
        lines.append("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dir(OUTPUT_DIR)
    _ensure_dir(PARTS_DIR)
    records: list[dict[str, Any]] = []
    for spec in _selected_tasks(_load_summary()):
        renderer = RENDERERS.get(spec.task_id)
        if renderer is None:
            records.append({"task_id": spec.task_id, "status": "missing_renderer"})
            continue
        result = renderer(spec)
        result["category"] = spec.category
        result["failed_metrics"] = spec.failed_metrics
        records.append(result)
    summary = {
        "selected_count": len(records),
        "rendered_count": sum(1 for item in records if item.get("status") == "rendered"),
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
