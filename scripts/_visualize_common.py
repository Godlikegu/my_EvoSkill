"""Shared helpers for operator-only reconstruction visualizations.

This module is intentionally outside the harness path.  Per-task
``visualize.py`` scripts may import it to avoid repeating plotting and metric
boilerplate.  It is for offline operator analysis only and may read hidden
ground-truth/reference files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def squeeze_first_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def ensure_2d_image(array: np.ndarray, *, prefer_last: bool = False) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.real_if_close(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return np.asarray(arr)
    if arr.ndim == 3:
        if prefer_last:
            return np.asarray(arr[..., arr.shape[-1] // 2])
        return np.asarray(arr[arr.shape[0] // 2])
    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size)))
        if side * side == arr.size:
            return np.asarray(arr.reshape(side, side))
    raise ValueError(f"cannot convert shape {list(arr.shape)} to a 2D image")


def cube_to_rgb(array: np.ndarray) -> np.ndarray:
    cube = np.asarray(array, dtype=np.float64)
    while cube.ndim > 3 and cube.shape[0] == 1:
        cube = cube[0]
    if cube.ndim != 3:
        raise ValueError(f"expected 3D cube, got shape {list(cube.shape)}")
    band_indices = [min(cube.shape[-1] - 1, idx) for idx in (24, cube.shape[-1] // 2, 6)]
    rgb = np.stack([cube[..., band_indices[0]], cube[..., band_indices[1]], cube[..., band_indices[2]]], axis=-1)
    rgb = rgb - np.nanmin(rgb)
    denom = np.nanmax(rgb)
    if denom > 0:
        rgb = rgb / denom
    return np.clip(rgb, 0.0, 1.0)


def ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    denom = np.linalg.norm(est) * np.linalg.norm(ref)
    if denom == 0:
        return 0.0
    return float(np.dot(est, ref) / denom)


def nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    dynamic_range = float(np.nanmax(ref) - np.nanmin(ref))
    if dynamic_range == 0:
        return float("inf")
    return float(np.sqrt(np.nanmean((est - ref) ** 2)) / dynamic_range)


def mae(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.nanmean(np.abs(np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))


def psnr(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mse = float(np.nanmean((est - ref) ** 2))
    if mse == 0:
        return float("inf")
    dynamic_range = float(np.nanmax(ref) - np.nanmin(ref))
    if dynamic_range == 0:
        dynamic_range = float(np.nanmax(np.abs(ref))) or 1.0
    return float(20.0 * np.log10(dynamic_range / np.sqrt(mse)))


def ssim_simple(estimate: np.ndarray, reference: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        ref = np.asarray(reference, dtype=np.float64)
        est = np.asarray(estimate, dtype=np.float64)
        data_range = float(np.nanmax(ref) - np.nanmin(ref)) or 1.0
        return float(structural_similarity(est, ref, data_range=data_range))
    except Exception:
        return float("nan")


def jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def render_image_triplet(
    *,
    gt: np.ndarray | None,
    baseline: np.ndarray | None,
    agent: np.ndarray,
    dest: Path,
    title: str,
    names: tuple[str, str, str] = ("Ground truth", "Baseline", "Agent"),
    notes: Iterable[str] = (),
    cmap: str = "gray",
) -> dict[str, Any]:
    """Render GT/baseline/agent and error maps into a 2x3 comparison PNG."""

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    agent_img = ensure_2d_image(agent)
    gt_img = ensure_2d_image(gt) if gt is not None else None
    baseline_img = ensure_2d_image(baseline) if baseline is not None else None

    metrics: dict[str, Any] = {}
    comparable = gt_img is not None and gt_img.shape == agent_img.shape
    if comparable:
        metrics.update(
            {
                "agent_ncc": ncc(agent_img, gt_img),
                "agent_nrmse": nrmse(agent_img, gt_img),
                "agent_mae": mae(agent_img, gt_img),
                "agent_psnr": psnr(agent_img, gt_img),
                "agent_ssim": ssim_simple(agent_img, gt_img),
            }
        )
    if comparable and baseline_img is not None and baseline_img.shape == gt_img.shape:
        metrics.update(
            {
                "baseline_ncc": ncc(baseline_img, gt_img),
                "baseline_nrmse": nrmse(baseline_img, gt_img),
                "baseline_mae": mae(baseline_img, gt_img),
                "baseline_psnr": psnr(baseline_img, gt_img),
                "baseline_ssim": ssim_simple(baseline_img, gt_img),
            }
        )

    panels: list[tuple[str, np.ndarray | None, str]] = [
        (names[0], gt_img, cmap),
        (names[1], baseline_img, cmap),
        (names[2], agent_img, cmap),
        ("", None, cmap),
        ("|Baseline - GT|", np.abs(baseline_img - gt_img) if baseline_img is not None and comparable else None, "hot"),
        ("|Agent - GT|", np.abs(agent_img - gt_img) if comparable else None, "hot"),
    ]

    if gt_img is not None:
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
    else:
        vmin = float(np.nanmin(agent_img))
        vmax = float(np.nanmax(agent_img))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (panel_title, image, panel_cmap) in zip(axes.ravel(), panels):
        ax.axis("off")
        if panel_title:
            ax.set_title(panel_title, fontsize=10)
        if image is not None:
            kwargs: dict[str, Any] = {"cmap": panel_cmap}
            if panel_cmap == cmap:
                kwargs.update({"vmin": vmin, "vmax": vmax})
            ax.imshow(image, **kwargs)

    note_text = " | ".join(note for note in notes if note)
    fig.suptitle(title, fontsize=14)
    if note_text:
        fig.text(0.5, 0.02, note_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if note_text else 0, 1, 0.96))
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return jsonable_metrics(metrics)


def render_curve_triplet(
    *,
    gt: np.ndarray | None,
    baseline: np.ndarray | None,
    agent: np.ndarray,
    dest: Path,
    title: str,
    names: tuple[str, str, str] = ("Ground truth", "Baseline", "Agent"),
    ylabel: str = "Value",
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    agent_curve = np.asarray(agent, dtype=np.float64).reshape(-1)
    gt_curve = np.asarray(gt, dtype=np.float64).reshape(-1) if gt is not None else None
    baseline_curve = np.asarray(baseline, dtype=np.float64).reshape(-1) if baseline is not None else None

    metrics: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if gt_curve is not None:
        ax.plot(np.arange(gt_curve.size), gt_curve, label=names[0], linewidth=2)
    if baseline_curve is not None:
        label = names[1]
        if gt_curve is not None and gt_curve.size == baseline_curve.size:
            metrics["baseline_mae"] = mae(baseline_curve, gt_curve)
            label = f"{label} (MAE={metrics['baseline_mae']:.4g})"
        ax.plot(np.arange(baseline_curve.size), baseline_curve, label=label, linewidth=2)
    label = names[2]
    if gt_curve is not None and gt_curve.size == agent_curve.size:
        metrics["agent_mae"] = mae(agent_curve, gt_curve)
        label = f"{label} (MAE={metrics['agent_mae']:.4g})"
    ax.plot(np.arange(agent_curve.size), agent_curve, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    note_text = " | ".join(note for note in notes if note)
    if note_text:
        fig.text(0.5, 0.01, note_text, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return jsonable_metrics(metrics)


def render_agent_overview(
    *,
    agent: np.ndarray,
    dest: Path,
    title: str,
    cmap: str = "gray",
) -> dict[str, Any]:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_2d_image(agent)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(image, cmap=cmap)
    axes[0].set_title("Agent output")
    axes[0].axis("off")
    axes[1].plot(image[image.shape[0] // 2, :])
    axes[1].set_title("Centre row")
    axes[1].grid(alpha=0.3)
    axes[2].hist(np.asarray(image).ravel(), bins=80)
    axes[2].set_title("Histogram")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"agent_min": float(np.nanmin(image)), "agent_max": float(np.nanmax(image))}
