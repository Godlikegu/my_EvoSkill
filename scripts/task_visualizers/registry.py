"""Operator-only, notebook-oriented task renderers.

The renderers in this module deliberately avoid a contract-generic fallback.
Each task is registered explicitly, with named reconstruction/reference fields
and a plotting layout chosen to match the task notebooks at a useful level of
detail.  These routines may read hidden ground truth and reference outputs; do
not call them from the agent sandbox.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RenderContext:
    task_id: str
    task_root: Path
    recon_path: Path
    output_dir: Path
    repo_root: Path
    run_id: str
    verdict: str


Renderer = Callable[[RenderContext], dict[str, Any]]


@dataclass(frozen=True)
class ArrayRef:
    relpath: str
    key: str | None = None
    label: str = ""


@dataclass(frozen=True)
class PanelSpec:
    output_key: str
    title: str
    gt: ArrayRef | None = None
    baseline: ArrayRef | None = None
    baseline_label: str = "Reference"
    mode: str = "auto"
    cmap: str = "gray"
    error: bool = True


@dataclass(frozen=True)
class TaskSpec:
    panels: tuple[PanelSpec, ...]
    notes: str = ""


def registered_task_ids() -> list[str]:
    return sorted(RENDERERS)


def get_renderer(task_id: str) -> Renderer | None:
    return RENDERERS.get(task_id)


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_array(task_root: Path, ref: ArrayRef) -> np.ndarray:
    path = task_root / ref.relpath
    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = _npz(path)
        if ref.key is None:
            if len(payload) != 1:
                raise KeyError(f"{path} has multiple arrays; a key is required")
            return next(iter(payload.values()))
        return np.asarray(payload[ref.key])
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))
    if suffix == ".mat":
        from scipy.io import loadmat

        payload = loadmat(path)
        if ref.key is None:
            candidates = [v for k, v in payload.items() if not k.startswith("__")]
            if len(candidates) != 1:
                raise KeyError(f"{path} has multiple mat arrays; a key is required")
            return np.asarray(candidates[0])
        return np.asarray(payload[ref.key])
    if suffix == ".csv":
        rows: list[list[float]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                vals: list[float] = []
                for cell in row:
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        pass
                if vals:
                    rows.append(vals)
        return np.asarray(rows, dtype=np.float64)
    raise ValueError(f"unsupported array source: {path}")


def _load_recon(ctx: RenderContext) -> dict[str, np.ndarray]:
    return _npz(ctx.recon_path)


def _real(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    out = np.real_if_close(out)
    if np.iscomplexobj(out):
        out = np.abs(out)
    return np.asarray(out, dtype=np.float64)


def _squeeze(arr: np.ndarray) -> np.ndarray:
    out = _real(arr)
    while out.ndim > 0 and out.shape[0] == 1:
        out = out[0]
    return out


def _image(arr: np.ndarray) -> np.ndarray:
    out = _squeeze(arr)
    if out.ndim == 0:
        return out.reshape(1, 1)
    if out.ndim == 1:
        side = int(round(math.sqrt(out.size)))
        if side * side == out.size:
            return out.reshape(side, side)
        return out.reshape(1, -1)
    while out.ndim > 3:
        out = out[out.shape[0] // 2]
    if out.ndim == 3:
        if out.shape[-1] in (3, 4):
            return out
        if out.shape[-1] <= 64 and out.shape[0] > out.shape[-1] and out.shape[1] > out.shape[-1]:
            return out[..., out.shape[-1] // 2]
        return out[out.shape[0] // 2]
    return out


def _curve(arr: np.ndarray) -> np.ndarray:
    return _real(arr).reshape(-1)


def _center_crop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a2 = _image(a)
    b2 = _image(b)
    if a2.shape == b2.shape:
        return a2, b2
    if a2.ndim != 2 or b2.ndim != 2:
        return a2, b2
    h = min(a2.shape[0], b2.shape[0])
    w = min(a2.shape[1], b2.shape[1])

    def crop(x: np.ndarray) -> np.ndarray:
        y0 = max(0, (x.shape[0] - h) // 2)
        x0 = max(0, (x.shape[1] - w) // 2)
        return x[y0 : y0 + h, x0 : x0 + w]

    return crop(a2), crop(b2)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _center_crop_pair(a, b)
    denom = float(np.nanmax(bb) - np.nanmin(bb)) or 1.0
    return float(np.sqrt(np.nanmean((aa - bb) ** 2)) / denom)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _center_crop_pair(a, b)
    va = aa.reshape(-1)
    vb = bb.reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _imshow(ax: Any, arr: np.ndarray, *, title: str, cmap: str = "gray", vmin: float | None = None, vmax: float | None = None) -> None:
    img = _image(arr)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = np.asarray(img, dtype=np.float64)
        rgb = rgb - np.nanmin(rgb)
        denom = float(np.nanmax(rgb)) or 1.0
        ax.imshow(np.clip(rgb / denom, 0.0, 1.0))
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)


def _save_panel_grid(
    dest: Path,
    panels: Sequence[tuple[str, np.ndarray, str]],
    *,
    title: str,
    same_scale_from: np.ndarray | None = None,
    cols: int = 4,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = min(cols, max(1, len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows), squeeze=False)
    vmin = vmax = None
    if same_scale_from is not None:
        base = _image(same_scale_from)
        if base.ndim == 2:
            vmin = float(np.nanmin(base))
            vmax = float(np.nanmax(base))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, arr, cmap) in zip(axes.ravel(), panels):
        use_scale = cmap == "gray"
        _imshow(ax, arr, title=label, cmap=cmap, vmin=vmin if use_scale else None, vmax=vmax if use_scale else None)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_error_grid(dest: Path, errors: Sequence[tuple[str, np.ndarray]], *, title: str) -> None:
    if not errors:
        return
    vmax = max(float(np.nanpercentile(np.abs(_image(err)), 99)) for _, err in errors) or None
    _save_panel_grid(dest, [(label, np.abs(err), "hot") for label, err in errors], title=title, cols=min(4, len(errors)))


def _save_curves(dest: Path, curves: Sequence[tuple[str, np.ndarray]], *, title: str, ylabel: str = "Value") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, values in curves:
        y = _curve(values)
        ax.plot(np.arange(y.size), y, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_bar(dest: Path, values: dict[str, dict[str, float]], *, title: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values)
    metrics = sorted({m for row in values.values() for m in row})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * max(1, len(metrics)), 4), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        y = [values[label].get(metric, np.nan) for label in labels]
        ax.bar(labels, y)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _rgb_cube(cube: np.ndarray) -> np.ndarray:
    arr = _squeeze(cube)
    if arr.ndim != 3:
        return _image(arr)
    if arr.shape[-1] in (3, 4):
        return arr
    idx = [min(arr.shape[-1] - 1, i) for i in (24, arr.shape[-1] // 2, 6)]
    rgb = np.stack([arr[..., idx[0]], arr[..., idx[1]], arr[..., idx[2]]], axis=-1)
    rgb = rgb - np.nanmin(rgb)
    denom = float(np.nanmax(rgb)) or 1.0
    return np.clip(rgb / denom, 0, 1)


def _render_spec(ctx: RenderContext, spec: TaskSpec) -> dict[str, Any]:
    recon = _load_recon(ctx)
    figures: list[str] = []
    metrics: dict[str, Any] = {}
    counter = 1
    for panel in spec.panels:
        if panel.output_key not in recon:
            metrics[f"{panel.output_key}_missing"] = True
            continue
        agent = recon[panel.output_key]
        gt = _load_array(ctx.task_root, panel.gt) if panel.gt else None
        baseline = _load_array(ctx.task_root, panel.baseline) if panel.baseline else None

        safe = panel.output_key.replace("/", "_")
        if panel.mode == "curve" or _squeeze(agent).ndim <= 1:
            curves: list[tuple[str, np.ndarray]] = []
            if gt is not None:
                curves.append(("Ground truth", gt))
            if baseline is not None:
                curves.append((panel.baseline_label, baseline))
            curves.append(("Agent", agent))
            dest = ctx.output_dir / f"{counter:02d}_{safe}.png"
            _save_curves(dest, curves, title=f"{ctx.task_id}: {panel.title}")
            figures.append(str(dest))
            counter += 1
            continue

        if panel.mode == "rgb":
            panels: list[tuple[str, np.ndarray, str]] = []
            if gt is not None:
                panels.append(("Ground truth RGB preview", _rgb_cube(gt), "gray"))
            if baseline is not None:
                panels.append((panel.baseline_label, _rgb_cube(baseline), "gray"))
            panels.append(("Agent RGB preview", _rgb_cube(agent), "gray"))
        else:
            panels = []
            if gt is not None:
                panels.append(("Ground truth", gt, panel.cmap))
            if baseline is not None:
                panels.append((panel.baseline_label, baseline, panel.cmap))
            panels.append(("Agent", agent, panel.cmap))

        dest = ctx.output_dir / f"{counter:02d}_{safe}_reconstruction.png"
        _save_panel_grid(dest, panels, title=f"{ctx.task_id}: {panel.title}", same_scale_from=gt if gt is not None else agent)
        figures.append(str(dest))
        counter += 1

        errors: list[tuple[str, np.ndarray]] = []
        if panel.error and gt is not None:
            if baseline is not None:
                b2, g2 = _center_crop_pair(baseline, gt)
                if b2.shape == g2.shape:
                    errors.append((f"|{panel.baseline_label} - GT|", b2 - g2))
            a2, g2 = _center_crop_pair(agent, gt)
            if a2.shape == g2.shape:
                errors.append(("|Agent - GT|", a2 - g2))
                metrics[f"{panel.output_key}_agent_ncc"] = _ncc(agent, gt)
                metrics[f"{panel.output_key}_agent_nrmse"] = _nrmse(agent, gt)
        if errors:
            dest = ctx.output_dir / f"{counter:02d}_{safe}_error_map.png"
            _save_error_grid(dest, errors, title=f"{ctx.task_id}: {panel.title} errors")
            figures.append(str(dest))
            counter += 1

    return {"figures": figures, "metrics": metrics, "renderer": "task_visualizers.registry"}


def _ct_poisson_lowdose(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)["phantom"]
    gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", "phantom"))
    raw = _npz(ctx.task_root / "data" / "raw_data.npz")
    ref = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "reconstructions.npz")
    figures: list[str] = []

    dest = ctx.output_dir / "01_sinograms.png"
    _save_panel_grid(
        dest,
        [
            ("Clean sinogram", raw["sinogram_clean"], "magma"),
            ("Low-dose sinogram", raw["sinogram_low_dose"], "magma"),
            ("High-dose sinogram", raw["sinogram_high_dose"], "magma"),
        ],
        title="ct_poisson_lowdose: notebook sinogram comparison",
        cols=3,
    )
    figures.append(str(dest))

    dest = ctx.output_dir / "02_reconstruction.png"
    panels = [
        ("Ground truth phantom", gt, "gray"),
        ("Unweighted reconstruction", ref["recon_unweighted"], "gray"),
        ("PWLS low-dose reference", ref["recon_pwls_low"], "gray"),
        ("Agent reconstruction", recon, "gray"),
    ]
    _save_panel_grid(dest, panels, title="ct_poisson_lowdose: GT / Unweighted / PWLS / Agent", same_scale_from=gt, cols=4)
    figures.append(str(dest))

    dest = ctx.output_dir / "03_error_map.png"
    _save_error_grid(
        dest,
        [
            ("|Unweighted - GT|", _image(ref["recon_unweighted"]) - _image(gt)),
            ("|PWLS low - GT|", _image(ref["recon_pwls_low"]) - _image(gt)),
            ("|Agent - GT|", _image(recon) - _image(gt)),
        ],
        title="ct_poisson_lowdose: error maps",
    )
    figures.append(str(dest))

    values = {
        "Unweighted": {"NCC": _ncc(ref["recon_unweighted"], gt), "NRMSE": _nrmse(ref["recon_unweighted"], gt)},
        "PWLS low": {"NCC": _ncc(ref["recon_pwls_low"], gt), "NRMSE": _nrmse(ref["recon_pwls_low"], gt)},
        "Agent": {"NCC": _ncc(recon, gt), "NRMSE": _nrmse(recon, gt)},
    }
    dest = ctx.output_dir / "04_metrics.png"
    _save_bar(dest, values, title="ct_poisson_lowdose: notebook-style reconstruction metrics")
    figures.append(str(dest))
    return {"figures": figures, "metrics": {"phantom_agent_ncc": values["Agent"]["NCC"], "phantom_agent_nrmse": values["Agent"]["NRMSE"]}, "renderer": "ct_poisson_lowdose"}


def _ct_sparse_view(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)["phantom"]
    gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", "phantom"))
    raw = _npz(ctx.task_root / "data" / "raw_data.npz")
    ref = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "reconstructions.npz")
    figures: list[str] = []

    dest = ctx.output_dir / "01_sinograms.png"
    _save_panel_grid(
        dest,
        [("Sparse-view sinogram", raw["sinogram_sparse"], "magma"), ("Full-view sinogram", raw["sinogram_full"], "magma")],
        title="ct_sparse_view: notebook sinograms",
        cols=2,
    )
    figures.append(str(dest))
    dest = ctx.output_dir / "02_reconstruction.png"
    _save_panel_grid(
        dest,
        [
            ("Ground truth phantom", gt, "gray"),
            ("Full-view FBP", ref["fbp_full"], "gray"),
            ("Sparse-view FBP", ref["fbp_sparse"], "gray"),
            ("TV reference", ref["tv_recon"], "gray"),
            ("Agent reconstruction", recon, "gray"),
        ],
        title="ct_sparse_view: notebook reconstruction comparison",
        same_scale_from=gt,
        cols=5,
    )
    figures.append(str(dest))
    dest = ctx.output_dir / "03_error_map.png"
    _save_error_grid(
        dest,
        [
            ("|Sparse FBP - GT|", _image(ref["fbp_sparse"]) - _image(gt)),
            ("|TV - GT|", _image(ref["tv_recon"]) - _image(gt)),
            ("|Agent - GT|", _image(recon) - _image(gt)),
        ],
        title="ct_sparse_view: error maps",
    )
    figures.append(str(dest))
    if "loss_history" in ref:
        dest = ctx.output_dir / "04_convergence.png"
        _save_curves(dest, [("TV loss", ref["loss_history"])], title="ct_sparse_view: TV convergence", ylabel="Loss")
        figures.append(str(dest))
    return {"figures": figures, "metrics": {"phantom_agent_ncc": _ncc(recon, gt), "phantom_agent_nrmse": _nrmse(recon, gt)}, "renderer": "ct_sparse_view"}


def _ct_fan_beam(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)["phantom"]
    gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", "phantom"))
    raw = _npz(ctx.task_root / "data" / "raw_data.npz")
    full = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "recon_fbp_full.npz")["reconstruction"]
    short = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "recon_fbp_short.npz")["reconstruction"]
    tv_payload = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "recon_tv_short.npz")
    tv = tv_payload["reconstruction"]
    figures: list[str] = []
    dest = ctx.output_dir / "01_sinograms.png"
    _save_panel_grid(dest, [("Full fan-beam sinogram", raw["sino_full"], "magma"), ("Short-scan sinogram", raw["sino_short"], "magma")], title="ct_fan_beam: notebook sinograms", cols=2)
    figures.append(str(dest))
    dest = ctx.output_dir / "02_reconstruction.png"
    _save_panel_grid(dest, [("Ground truth", gt, "gray"), ("Full FBP", full, "gray"), ("Short FBP", short, "gray"), ("Short TV", tv, "gray"), ("Agent", recon, "gray")], title="ct_fan_beam: reconstruction comparison", same_scale_from=gt, cols=5)
    figures.append(str(dest))
    dest = ctx.output_dir / "03_error_map.png"
    _save_error_grid(dest, [("|Short FBP - GT|", _image(short) - _image(gt)), ("|Short TV - GT|", _image(tv) - _image(gt)), ("|Agent - GT|", _image(recon) - _image(gt))], title="ct_fan_beam: error maps")
    figures.append(str(dest))
    if "loss_history" in tv_payload:
        dest = ctx.output_dir / "04_convergence.png"
        _save_curves(dest, [("TV loss", tv_payload["loss_history"])], title="ct_fan_beam: TV convergence", ylabel="Loss")
        figures.append(str(dest))
    return {"figures": figures, "metrics": {"phantom_agent_ncc": _ncc(recon, gt), "phantom_agent_nrmse": _nrmse(recon, gt)}, "renderer": "ct_fan_beam"}


def _mri_comparison(
    ctx: RenderContext,
    *,
    output_key: str,
    gt_key: str,
    references: Sequence[tuple[str, str, str]],
    title: str,
    raw_mask_key: str | None = None,
) -> dict[str, Any]:
    recon = _load_recon(ctx)[output_key]
    gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", gt_key))
    figures: list[str] = []
    panels: list[tuple[str, np.ndarray, str]] = [("Ground truth", gt, "gray")]
    errors: list[tuple[str, np.ndarray]] = []
    for label, relpath, key in references:
        arr = _load_array(ctx.task_root, ArrayRef(relpath, key))
        panels.append((label, arr, "gray"))
        r2, g2 = _center_crop_pair(arr, gt)
        if r2.shape == g2.shape:
            errors.append((f"|{label} - GT|", r2 - g2))
    panels.append(("Agent reconstruction", recon, "gray"))
    a2, g2 = _center_crop_pair(recon, gt)
    if a2.shape == g2.shape:
        errors.append(("|Agent - GT|", a2 - g2))
    dest = ctx.output_dir / "01_reconstruction.png"
    _save_panel_grid(dest, panels, title=title, same_scale_from=gt, cols=min(5, len(panels)))
    figures.append(str(dest))
    dest = ctx.output_dir / "02_error_map.png"
    _save_error_grid(dest, errors, title=f"{title}: error maps")
    figures.append(str(dest))
    if raw_mask_key is not None:
        raw = _npz(ctx.task_root / "data" / "raw_data.npz")
        if raw_mask_key in raw:
            dest = ctx.output_dir / "03_sampling_mask.png"
            _save_panel_grid(dest, [("Notebook sampling/undersampling mask", raw[raw_mask_key], "gray")], title=f"{ctx.task_id}: sampling mask", cols=1)
            figures.append(str(dest))
    return {"figures": figures, "metrics": {f"{output_key}_agent_ncc": _ncc(recon, gt), f"{output_key}_agent_nrmse": _nrmse(recon, gt)}, "renderer": ctx.task_id}


def _mri_sense(ctx: RenderContext) -> dict[str, Any]:
    return _mri_comparison(
        ctx,
        output_key="image",
        gt_key="image",
        references=[("Zero-fill", "evaluation/reference_outputs/zerofill.npz", "reconstruction"), ("SENSE reference", "evaluation/reference_outputs/sense_reconstruction.npz", "reconstruction")],
        title="mri_sense: notebook reconstruction comparison",
    )


def _mri_varnet(ctx: RenderContext) -> dict[str, Any]:
    return _mri_comparison(
        ctx,
        output_key="reconstruction",
        gt_key="image",
        references=[("Zero-fill", "evaluation/reference_outputs/zerofill.npz", "reconstruction"), ("VarNet reference", "evaluation/reference_outputs/varnet_reconstruction.npz", "reconstruction")],
        title="mri_varnet: notebook reconstruction comparison",
    )


def _mri_l1_wavelet(ctx: RenderContext) -> dict[str, Any]:
    return _mri_comparison(
        ctx,
        output_key="phantom",
        gt_key="phantom",
        references=[("L1 wavelet reference", "evaluation/reference_outputs/l1_wavelet_reconstruction.npz", "reconstruction"), ("TV reference", "evaluation/reference_outputs/tv_reconstruction.npz", "reconstruction")],
        title="mri_l1_wavelet: notebook reconstruction grid",
        raw_mask_key="undersampling_mask",
    )


def _cars_spectroscopy(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)
    gt = _npz(ctx.task_root / "data" / "ground_truth.npz")
    raw = _npz(ctx.task_root / "data" / "raw_data.npz")
    ref = _npz(ctx.task_root / "evaluation" / "reference_outputs" / "reconstruction.npz")
    figures: list[str] = []
    dest = ctx.output_dir / "01_spectrum_fit.png"
    _save_curves(
        dest,
        [
            ("Ground-truth spectrum", gt["spectrum"]),
            ("Noisy measurement", raw["measurements"]),
            ("Reference fit", ref["y_pred"]),
            ("Agent reconstructed spectrum", recon["reconstructed_spectrum"]),
        ],
        title="cars_spectroscopy: notebook spectrum fit",
        ylabel="Intensity",
    )
    figures.append(str(dest))
    values = {
        "Ground truth": {"Temperature K": float(np.asarray(gt["temperature"]).reshape(-1)[0])},
        "Reference": {"Temperature K": float(np.asarray(ref["temperature_pred"]).reshape(-1)[0])},
        "Agent": {"Temperature K": float(np.asarray(recon["estimated_temperature_K"]).reshape(-1)[0])},
    }
    dest = ctx.output_dir / "02_temperature.png"
    _save_bar(dest, values, title="cars_spectroscopy: inferred gas temperature")
    figures.append(str(dest))
    return {"figures": figures, "metrics": {"temperature_agent_K": values["Agent"]["Temperature K"]}, "renderer": "cars_spectroscopy"}


def _era5_tensorvar(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)["state"]
    gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", "state"))
    ref = _load_array(ctx.task_root, ArrayRef("evaluation/reference_outputs/trajectory.npy"))
    figures: list[str] = []
    for ti in [0, min(2, _squeeze(recon).shape[0] - 1), _squeeze(recon).shape[0] - 1]:
        panels = []
        for ch in range(min(5, _squeeze(recon).shape[1])):
            panels.append((f"GT t={ti} ch={ch}", _squeeze(gt)[ti, ch], "coolwarm"))
            panels.append((f"Agent t={ti} ch={ch}", _squeeze(recon)[ti, ch], "coolwarm"))
        dest = ctx.output_dir / f"{len(figures)+1:02d}_state_t{ti}.png"
        _save_panel_grid(dest, panels, title=f"era5_tensorvar: notebook trajectory channels at t={ti}", cols=5)
        figures.append(str(dest))
    dest = ctx.output_dir / f"{len(figures)+1:02d}_state_error.png"
    _save_error_grid(dest, [("Reference trajectory mean error", _squeeze(ref)[: _squeeze(recon).shape[0]].mean(axis=1)[0] - _squeeze(gt).mean(axis=1)[0]), ("Agent trajectory mean error", _squeeze(recon).mean(axis=1)[0] - _squeeze(gt).mean(axis=1)[0])], title="era5_tensorvar: trajectory error preview")
    figures.append(str(dest))
    return {"figures": figures, "metrics": {"state_agent_nrmse": _nrmse(recon, gt)}, "renderer": "era5_tensorvar"}


def _hessian_sim(ctx: RenderContext) -> dict[str, Any]:
    recon = _load_recon(ctx)["data"]
    refs = {
        "Wiener": _load_array(ctx.task_root, ArrayRef("evaluation/reference_outputs/wiener_sim.npz", "data")),
        "Hessian": _load_array(ctx.task_root, ArrayRef("evaluation/reference_outputs/hessian_sim.npz", "data")),
        "TV": _load_array(ctx.task_root, ArrayRef("evaluation/reference_outputs/tv_sim.npz", "data")),
    }
    figures: list[str] = []
    recon_squeezed = _squeeze(recon)
    refs = {name: _squeeze(arr) for name, arr in refs.items()}
    mid = recon_squeezed.shape[0] // 2
    dest = ctx.output_dir / "01_sim_planes.png"
    panels = [(f"{name} plane {mid}", arr[mid], "gray") for name, arr in refs.items()]
    panels.append(("Agent plane", recon_squeezed[mid], "gray"))
    _save_panel_grid(dest, panels, title="hessian_sim: notebook SIM reconstruction planes", cols=4)
    figures.append(str(dest))
    dest = ctx.output_dir / "02_fft_preview.png"
    fft_panels = []
    for name, arr in [*refs.items(), ("Agent", recon_squeezed)]:
        img = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(_image(arr[mid])))))
        fft_panels.append((f"{name} FFT", img, "magma"))
    _save_panel_grid(dest, fft_panels, title="hessian_sim: Fourier previews", cols=4)
    figures.append(str(dest))
    center = recon_squeezed.shape[-2] // 2
    dest = ctx.output_dir / "03_line_profiles.png"
    curves = [(name, _image(arr[mid])[center, :]) for name, arr in refs.items()]
    curves.append(("Agent", _image(recon_squeezed[mid])[center, :]))
    _save_curves(dest, curves, title="hessian_sim: notebook center line profiles")
    figures.append(str(dest))
    return {"figures": figures, "metrics": {}, "renderer": "hessian_sim"}


def _sequence_spec(output_key: str, gt_key: str, title: str, ref: ArrayRef | None = None) -> Renderer:
    def render(ctx: RenderContext) -> dict[str, Any]:
        recon = _load_recon(ctx)[output_key]
        gt = _load_array(ctx.task_root, ArrayRef("data/ground_truth.npz", gt_key))
        baseline = _load_array(ctx.task_root, ref) if ref is not None else None
        arr = _squeeze(recon)
        idxs = [0, arr.shape[0] // 2, arr.shape[0] - 1] if arr.ndim >= 3 else [0]
        figures: list[str] = []
        panels: list[tuple[str, np.ndarray, str]] = []
        for i in dict.fromkeys(idxs):
            panels.append((f"GT frame {i}", _squeeze(gt)[i], "gray"))
            if baseline is not None:
                panels.append((f"Reference frame {i}", _squeeze(baseline)[i], "gray"))
            panels.append((f"Agent frame {i}", arr[i], "gray"))
        dest = ctx.output_dir / "01_sequence_frames.png"
        _save_panel_grid(dest, panels, title=title, same_scale_from=gt, cols=3 if baseline is not None else 2)
        figures.append(str(dest))
        return {"figures": figures, "metrics": {f"{output_key}_agent_nrmse": _nrmse(recon, gt)}, "renderer": ctx.task_id}

    return render


def _scalar_spec(keys: Sequence[str], gt_rel: str = "data/ground_truth.npz") -> Renderer:
    def render(ctx: RenderContext) -> dict[str, Any]:
        recon = _load_recon(ctx)
        gt = _npz(ctx.task_root / gt_rel) if (ctx.task_root / gt_rel).exists() else {}
        rows: dict[str, dict[str, float]] = {}
        for key in keys:
            if key in recon:
                rows[f"Agent {key}"] = {"value": float(np.asarray(recon[key]).reshape(-1)[0])}
            if key in gt:
                rows[f"GT {key}"] = {"value": float(np.asarray(gt[key]).reshape(-1)[0])}
        dest = ctx.output_dir / "01_scalar_outputs.png"
        _save_bar(dest, rows, title=f"{ctx.task_id}: notebook scalar outputs")
        return {"figures": [str(dest)], "metrics": {}, "renderer": ctx.task_id}

    return render


def _make_spec(spec: TaskSpec) -> Renderer:
    return lambda ctx: _render_spec(ctx, spec)


SPECS: dict[str, TaskSpec] = {
    "SSNP_ODT": TaskSpec((PanelSpec("delta_n", "refractive-index volume slices", ArrayRef("data/ground_truth.npz", "delta_n"), None),)),
    "confocal-nlos-fk": TaskSpec((PanelSpec("fk", "FK volume reconstruction", None, ArrayRef("evaluation/reference_outputs/reconstruction.npz", "fk"), "Reference FK"),)),
    "conventional_ptychography": TaskSpec((PanelSpec("object", "complex object magnitude/phase", ArrayRef("data/ground_truth.npz", "object"), None),)),
    "ct_dual_energy": TaskSpec((PanelSpec("tissue_map", "tissue map", ArrayRef("data/ground_truth.npz", "tissue_map"), ArrayRef("evaluation/reference_outputs/reference_reconstruction.npz", "tissue_map"), "Reference"), PanelSpec("bone_map", "bone map", ArrayRef("data/ground_truth.npz", "bone_map"), ArrayRef("evaluation/reference_outputs/reference_reconstruction.npz", "bone_map"), "Reference"))),
    "diffusion_mri_dti": TaskSpec((PanelSpec("fa_map", "FA map", ArrayRef("data/ground_truth.npz", "fa_map"), ArrayRef("evaluation/reference_outputs/dti_wls.npz", "fa_map"), "WLS DTI"), PanelSpec("md_map", "MD map", ArrayRef("data/ground_truth.npz", "md_map"), ArrayRef("evaluation/reference_outputs/dti_wls.npz", "md_map"), "WLS DTI"))),
    "eht_black_hole_UQ": TaskSpec((PanelSpec("posterior_mean", "posterior mean image", ArrayRef("data/ground_truth.npz", "image"), ArrayRef("evaluation/reference_outputs/posterior_mean.npy"), "Posterior mean"),)),
    "eht_black_hole_dynamic": TaskSpec((PanelSpec("reconstruction", "dynamic image sequence", ArrayRef("data/ground_truth.npz", "images"), ArrayRef("evaluation/reference_outputs/starwarps_reconstruction.npy"), "StarWarps reference"),)),
    "eht_black_hole_original": TaskSpec((PanelSpec("image", "black-hole image", ArrayRef("data/ground_truth.npz", "image"), ArrayRef("evaluation/reference_outputs/vis_rml_cal.npy"), "Vis RML calibrated"),)),
    "eht_black_hole_tomography": TaskSpec((PanelSpec("emission_3d", "3D emission volume", ArrayRef("data/ground_truth.npz", "emission_3d"), ArrayRef("evaluation/reference_outputs/emission_3d.npy"), "Reference volume"),)),
    "eit_conductivity_reconstruction": TaskSpec((PanelSpec("bp_perm_anomaly", "BP conductivity anomaly", ArrayRef("data/ground_truth.npz", "bp_perm_anomaly"), ArrayRef("evaluation/reference_outputs/reconstruction_bp.npy"), "BP reference", mode="curve"),)),
    "electron_ptychography": TaskSpec((PanelSpec("ptycho_phase", "ptychography phase", None, ArrayRef("data/baseline_reference.npz", "ptycho_phase"), "Baseline phase"),)),
    "exoplanet_imaging": TaskSpec((PanelSpec("klip_image", "KLIP residual image", None, ArrayRef("data/baseline_reference.npz", "klip_image"), "Baseline KLIP"),)),
    "fourier_ptychography": TaskSpec((PanelSpec("object", "Fourier ptychography object", ArrayRef("data/ground_truth.npz", "object"), None),)),
    "fpm_inr_reconstruction": TaskSpec((PanelSpec("amplitude_stack", "through-focus amplitude stack", ArrayRef("data/ground_truth.npz", "I_stack"), ArrayRef("evaluation/reference_outputs/aif_pred.npy"), "AIF reference"),)),
    "insar_phase_unwrapping": TaskSpec((PanelSpec("unwrapped_phase", "unwrapped phase", None, ArrayRef("data/baseline_reference.npz", "unwrapped_phase"), "Baseline unwrapped phase"),)),
    "lensless_imaging": TaskSpec((PanelSpec("reconstruction", "lensless RGB reconstruction", None, ArrayRef("data/baseline_reference.npz", "reconstruction"), "Baseline reconstruction"),)),
    "light_field_microscope": TaskSpec((PanelSpec("rl_reconstructions", "RL reconstructions", ArrayRef("data/ground_truth.npz", "target_images"), ArrayRef("evaluation/reference_outputs/baseline_reference.npz", "rl_reconstructions"), "RL reference"),)),
    "lucky_imaging": TaskSpec((PanelSpec("stacked", "lucky-imaging stacked frame", None, ArrayRef("data/baseline_reference.npz", "stacked"), "Baseline stacked"),)),
    "mcr_hyperspectral": TaskSpec((PanelSpec("concentrations_ravel", "MCR concentration maps", ArrayRef("data/ground_truth.npz", "concentrations_ravel"), None, mode="curve"),)),
    "microscope_denoising": TaskSpec((PanelSpec("denoised", "denoised microscopy image", None, ArrayRef("data/baseline_reference.npz", "denoised"), "Reference denoised"), PanelSpec("deconvolved", "deconvolved microscopy image", None, ArrayRef("data/baseline_reference.npz", "deconvolved"), "Reference deconvolved"))),
    "mri_dynamic_dce": TaskSpec((PanelSpec("dynamic_images", "DCE dynamic frames", ArrayRef("data/ground_truth.npz", "dynamic_images"), ArrayRef("evaluation/reference_outputs/tv_reconstruction.npz", "reconstruction"), "TV reference"),)),
    "mri_grappa": TaskSpec((PanelSpec("reconstruction", "GRAPPA reconstruction", ArrayRef("data/ground_truth.npz", "image"), ArrayRef("evaluation/reference_outputs/grappa_reconstruction.npz", "reconstruction"), "GRAPPA reference"),)),
    "mri_noncartesian_cs": TaskSpec((PanelSpec("phantom", "non-Cartesian CS reconstruction", ArrayRef("data/ground_truth.npz", "phantom"), ArrayRef("evaluation/reference_outputs/l1wav_reconstruction.npz", "reconstruction"), "L1 wavelet reference"),)),
    "mri_pnp_admm": TaskSpec((PanelSpec("image", "PnP-ADMM MRI image", ArrayRef("data/ground_truth.npz", "image"), ArrayRef("evaluation/reference_outputs/pnp_admm_reconstruction.npz", "reconstruction"), "PnP-ADMM reference"),)),
    "mri_t2_mapping": TaskSpec((PanelSpec("T2_map", "T2 map", ArrayRef("data/ground_truth.npz", "T2_map"), ArrayRef("evaluation/reference_outputs/T2_map_nonlinear.npz", "T2_map"), "Nonlinear reference"), PanelSpec("M0_map", "M0 map", ArrayRef("data/ground_truth.npz", "M0_map"), ArrayRef("evaluation/reference_outputs/T2_map_nonlinear.npz", "M0_map"), "Nonlinear reference"))),
    "mri_tv": TaskSpec((PanelSpec("mvue", "TV MRI reconstruction", ArrayRef("data/ground_truth.npz", "mvue"), ArrayRef("evaluation/reference_outputs/tv_reconstruction.npz", "reconstruction"), "TV reference"),)),
    "pet_mlem": TaskSpec((PanelSpec("activity_map", "PET activity map", ArrayRef("data/ground_truth.npz", "activity_map"), ArrayRef("evaluation/reference_outputs/recon_mlem.npz", "reconstruction"), "MLEM reference"),)),
    "photoacoustic_tomography": TaskSpec((PanelSpec("reconstruction", "photoacoustic image", ArrayRef("data/ground_truth.npz", "ground_truth_image"), ArrayRef("evaluation/reference_outputs/reconstruction.npz", "reconstruction"), "Reference reconstruction"),)),
    "plane_wave_ultrasound": TaskSpec((PanelSpec("bmode_fibers", "B-mode fibers", None, ArrayRef("data/baseline_reference.npz", "bmode_fibers"), "Baseline B-mode"), PanelSpec("bmode_cysts", "B-mode cysts", None, ArrayRef("data/baseline_reference.npz", "bmode_cysts"), "Baseline B-mode"))),
    "pnp_mri_reconstruction": TaskSpec((PanelSpec("img", "PnP MRI reconstruction", ArrayRef("data/ground_truth.npz", "img"), ArrayRef("evaluation/reference_outputs/pnp_mssn_recon.npy"), "PnP-MSSN reference"),)),
    "raman_cell_phenotyping": TaskSpec((PanelSpec("abundance_lipids", "lipid abundance", None, ArrayRef("data/baseline_reference.npz", "abundance_lipids"), "Baseline"), PanelSpec("abundance_nucleus", "nucleus abundance", None, ArrayRef("data/baseline_reference.npz", "abundance_nucleus"), "Baseline"), PanelSpec("abundance_cytoplasm", "cytoplasm abundance", None, ArrayRef("data/baseline_reference.npz", "abundance_cytoplasm"), "Baseline"), PanelSpec("abundance_background", "background abundance", None, ArrayRef("data/baseline_reference.npz", "abundance_background"), "Baseline"))),
    "reflection_ODT": TaskSpec((PanelSpec("delta_n", "reflection ODT delta-n slices", ArrayRef("data/ground_truth.npz", "delta_n"), ArrayRef("evaluation/reference_outputs/reconstruction.npy"), "Reference reconstruction"),)),
    "s2ism": TaskSpec((PanelSpec("reconstruction", "S2ISM reconstruction channels", ArrayRef("data/ground_truth.npz", "ground_truth"), ArrayRef("evaluation/reference_outputs/reconstruction.npz", "reconstruction"), "Reference"),)),
    "seismic_FWI_original": TaskSpec((PanelSpec("v_true", "FWI velocity model", ArrayRef("data/ground_truth.npz", "v_true"), ArrayRef("evaluation/reference_outputs/v_inv.npy"), "FWI reference"),)),
    "seismic_lsrtm_original": TaskSpec((PanelSpec("v_true", "LSRTM velocity / scatter image", ArrayRef("data/ground_truth.npz", "v_true"), ArrayRef("evaluation/reference_outputs/reference_reconstruction.npz", "scatter"), "Reference scatter"),)),
    "seismic_traveltime_tomography": TaskSpec((PanelSpec("velocity_perturbation", "velocity perturbation", ArrayRef("data/ground_truth.npz", "velocity"), ArrayRef("evaluation/reference_outputs/baseline_reference.npz", "velocity_perturbation"), "Baseline perturbation"),)),
    "shack-hartmann": TaskSpec((PanelSpec("reconstructed_phases", "wavefront phase levels", ArrayRef("data/ground_truth.npz", "wavefront_phases"), ArrayRef("evaluation/reference_outputs/reconstruction.npz", "reconstructed_phases"), "Reference reconstruction"),)),
    "shapelet_source_reconstruction": TaskSpec((PanelSpec("source_image", "source reconstruction", ArrayRef("data/ground_truth.npz", "source_image"), ArrayRef("evaluation/reference_outputs/lensing_outputs.npz", "source_recon_2d"), "Lensing reference"),)),
    "single_molecule_light_field": TaskSpec((PanelSpec("n_locs_3d_filtered", "localization scalar summary", None, ArrayRef("evaluation/reference_outputs/locs_3d.npz", "locs_3d"), "Reference localizations", mode="curve"),)),
    "spectral_snapshot_compressive_imaging": TaskSpec((PanelSpec("hyperspectral_cube", "hyperspectral pseudo-RGB", ArrayRef("data/ground_truth.npz", "hyperspectral_cube"), ArrayRef("evaluation/reference_outputs/kaist_crop256_01_result.mat", "img"), "Reference reconstruction", mode="rgb"),)),
    "ultrasound_sos_tomography": TaskSpec((PanelSpec("sos_phantom", "speed-of-sound map", ArrayRef("data/ground_truth.npz", "sos_phantom"), ArrayRef("evaluation/reference_outputs/reconstructions.npz", "sos_tv"), "TV reference"),)),
    "usct_FWI": TaskSpec((PanelSpec("vp_reconstructed", "USCT velocity reconstruction", None, ArrayRef("data/baseline_reference.npz", "vp_reconstructed"), "Baseline velocity"),)),
    "weather_radar_data_assimilation": TaskSpec((PanelSpec("reconstructed_frames", "radar target frames", ArrayRef("data/ground_truth.npz", "target_frames"), ArrayRef("evaluation/reference_outputs/reconstruction.npz", "reconstructed_frames"), "Reference reconstruction"),)),
    "xray_laminography_tike": TaskSpec((PanelSpec("volume", "x-ray laminography volume", ArrayRef("data/ground_truth.npz", "volume"), ArrayRef("evaluation/reference_outputs/reconstructed_volume.npy"), "Reference volume"),)),
    "xray_ptychography_tike": TaskSpec((PanelSpec("object_phase", "x-ray ptychography phase", None, ArrayRef("data/baseline_reference.npz", "object_phase"), "Baseline object phase"),)),
    "xray_tooth_gridrec": TaskSpec((PanelSpec("reconstruction", "x-ray tooth gridrec slices", None, ArrayRef("data/baseline_reference.npz", "reconstruction"), "Baseline reconstruction"),)),
}


RENDERERS: dict[str, Renderer] = {
    **{task_id: _make_spec(spec) for task_id, spec in SPECS.items()},
    "cars_spectroscopy": _cars_spectroscopy,
    "ct_poisson_lowdose": _ct_poisson_lowdose,
    "ct_sparse_view": _ct_sparse_view,
    "ct_fan_beam": _ct_fan_beam,
    "differentiable_deflectometry": _scalar_spec(("surface_0_roc_mm", "surface_1_roc_mm", "thickness_mm")),
    "eht_black_hole_feature_extraction_dynamic": _make_spec(TaskSpec((PanelSpec("position_angle_deg", "position-angle trajectory", ArrayRef("data/ground_truth.npz", "position_angle_deg"), None, mode="curve"),))),
    "era5_tensorvar": _era5_tensorvar,
    "hessian_sim": _hessian_sim,
    "mri_sense": _mri_sense,
    "mri_varnet": _mri_varnet,
    "mri_l1_wavelet": _mri_l1_wavelet,
}
