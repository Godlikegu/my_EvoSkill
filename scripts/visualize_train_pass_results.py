"""Visualize the six PASS wave-optics train results.

The figures intentionally mirror the plotting style used in each task's
notebook: SSNP slices/profiles, ptychography phase and diffraction panels,
confocal NLOS projections, seismic velocity/gather plots, ultrasound SoS
maps/profiles, and FWI velocity/error plots.

This is an operator-side visualization utility. It reads train task reference
data for side-by-side notebook-style comparisons, but it does not read or
write any valid-split task assets.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT.parent / "tasks"
LOG_ROOT = REPO_ROOT / "artifacts" / "logs" / "Vendor2_Claude-4.6-opus"
OUT_DIR = REPO_ROOT / "artifacts" / "visualizations" / "train_pass_6"


@dataclass(frozen=True)
class PassRun:
    task_id: str
    run_id: str


PASS_RUNS = (
    PassRun("SSNP_ODT", "run-1777700376-d10275"),
    PassRun("conventional_ptychography", "run-1777650205-13e8fd"),
    PassRun("confocal-nlos-fk", "run-1777697461-66431a"),
    PassRun("seismic_lsrtm_original", "run-1777365437-b08375"),
    PassRun("ultrasound_sos_tomography", "run-1777653310-f1ac36"),
    PassRun("seismic_FWI_original", "run-1777707253-e82211"),
)


def _load_summary(run: PassRun) -> dict:
    path = LOG_ROOT / run.task_id / run.run_id / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace(run: PassRun) -> Path:
    return Path(_load_summary(run)["workspace_root"])


def _output_npz(run: PassRun) -> Path:
    summary = _load_summary(run)
    rel = summary.get("primary_output_path") or summary.get("policy", {}).get(
        "primary_output_rel", "output/reconstruction.npz"
    )
    path = Path(rel)
    return path if path.is_absolute() else Path(summary["workspace_root"]) / path


def _task_data(run: PassRun, rel: str) -> Path:
    return TASK_ROOT / run.task_id / rel


def _save(fig: plt.Figure, name: str, created: list[str], *, dpi: int = 160) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    created.append(name)


def _squeeze_first(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else 0.0


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    rng = y.max() - y.min()
    return float(np.sqrt(np.mean((x - y) ** 2)) / (rng + 1e-12))


def _complex_to_hsv(arr: np.ndarray, max_amp: float | None = None) -> np.ndarray:
    arr2d = np.squeeze(arr)
    amp = np.abs(arr2d)
    phase = np.angle(arr2d)
    if max_amp is None:
        max_amp = float(amp.max()) + 1e-12
    hue = (phase + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), np.clip(amp / max_amp, 0, 1)], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def _norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def visualize_ssnp(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    meas = _squeeze_first(np.load(ws / "data" / "raw_data.npz")["measurements"])
    recon = _squeeze_first(np.load(_output_npz(run))["delta_n"])
    gt = _squeeze_first(np.load(_task_data(run, "data/ground_truth.npz"))["delta_n"])

    n_angles = meas.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.ravel()):
        if i < n_angles:
            theta = 360 * i / n_angles
            ax.imshow(meas[i], cmap="gray", origin="lower")
            ax.set_title(f"Angle {i} ({theta:.0f} deg)", fontsize=10)
        ax.axis("off")
    fig.suptitle("SSNP-IDT intensity measurements", fontsize=13)
    fig.tight_layout()
    _save(fig, "01_ssnp_measurements.png", created)

    nz, ny, nx = gt.shape
    z_mid = nz // 2
    vmax = max(np.percentile(gt, 99.5), np.percentile(recon, 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(gt[z_mid], cmap="hot", origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title(f"Ground Truth (z={z_mid})")
    axes[1].imshow(recon[z_mid], cmap="hot", origin="lower", vmin=0, vmax=vmax)
    axes[1].set_title(f"Reconstruction (z={z_mid})")
    im = axes[2].imshow(np.abs(gt[z_mid] - recon[z_mid]), cmap="hot", origin="lower")
    axes[2].set_title("|Difference|")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"SSNP-IDT center XY comparison  NCC={_ncc(recon, gt):.4f}  NRMSE={_nrmse(recon, gt):.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "02_ssnp_xy_comparison.png", created)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt[:, ny // 2, :], cmap="hot", origin="lower", vmin=0, vmax=vmax, aspect="auto")
    axes[0].set_title(f"Ground Truth XZ (y={ny//2})")
    axes[1].imshow(recon[:, ny // 2, :], cmap="hot", origin="lower", vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title(f"Reconstruction XZ (y={ny//2})")
    im = axes[2].imshow(np.abs(gt[:, ny // 2, :] - recon[:, ny // 2, :]), cmap="hot", origin="lower", aspect="auto")
    axes[2].set_title("|XZ Difference|")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("SSNP-IDT XZ cross-section comparison", fontsize=12)
    fig.tight_layout()
    _save(fig, "03_ssnp_xz_comparison.png", created)


def visualize_ptychography(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    ptychogram = raw["ptychogram"]
    encoder = raw["encoder"]
    obj = np.load(_output_npz(run))["object"]
    gt = np.load(_task_data(run, "data/ground_truth.npz"))["object"]
    meta = json.loads((ws / "data" / "meta_data.json").read_text(encoding="utf-8"))
    num_pos, nd = ptychogram.shape[0], ptychogram.shape[1]
    no = int(meta.get("No", obj.shape[0]))
    dxp = float(meta.get("dxp_m", 1.0))
    wavelength = float(meta.get("wavelength_m", 0.0))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    axes[0].imshow(np.log(ptychogram.mean(axis=0) + 1), cmap="hot")
    axes[0].set_title("mean (log)", fontsize=9)
    axes[0].axis("off")
    peak_idx = int(np.argmax(ptychogram.max(axis=(1, 2))))
    axes[1].imshow(np.log(ptychogram[peak_idx] + 1), cmap="hot")
    axes[1].set_title(f"peak intensity (#{peak_idx}, log)", fontsize=9)
    axes[1].axis("off")
    for i, idx in enumerate(np.linspace(0, num_pos - 1, 8, dtype=int)):
        ax = axes[i + 2]
        ax.imshow(np.log(ptychogram[idx] + 1), cmap="hot")
        ry, rx = encoder[idx, 0] * 1e6, encoder[idx, 1] * 1e6
        ax.set_title(f"pos #{idx}\n({rx:.0f},{ry:.0f}) um", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"CP diffraction patterns ({nd}x{nd} px, lambda={wavelength*1e9:.0f} nm, log scale)", fontsize=12)
    fig.tight_layout()
    _save(fig, "04_cp_diffraction_patterns.png", created)

    positions = (np.round(encoder / dxp) + no // 2 - nd // 2).astype(int)
    centers = positions + nd // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(centers[:, 1], centers[:, 0], s=20, alpha=0.7, c=np.arange(num_pos), cmap="viridis")
    axes[0].set_xlim(0, no)
    axes[0].set_ylim(no, 0)
    axes[0].set_aspect("equal")
    axes[0].set_title("Scan positions (pixel coordinates)")
    axes[0].set_xlabel("col [px]")
    axes[0].set_ylabel("row [px]")
    fig.colorbar(sc0, ax=axes[0], label="scan order")
    sc1 = axes[1].scatter(encoder[:, 1] * 1e6, encoder[:, 0] * 1e6, s=20, alpha=0.7, c=np.arange(num_pos), cmap="viridis")
    axes[1].set_aspect("equal")
    axes[1].set_title("Scan positions (physical coordinates)")
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    fig.colorbar(sc1, ax=axes[1], label="scan order")
    fig.suptitle(f"Fermat Spiral Scan Grid ({num_pos} positions)", fontsize=12)
    fig.tight_layout()
    _save(fig, "05_cp_scan_grid.png", created)

    gt_phase = np.angle(gt) - np.angle(gt).mean()
    obj_phase = np.angle(obj) - np.angle(obj).mean()
    phase_diff = obj_phase - gt_phase
    phase_ncc = _ncc(obj_phase, gt_phase)
    phase_nrmse = _nrmse(obj_phase, gt_phase)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    im = axes[0, 0].imshow(gt_phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title("GT phase (mean-subtracted)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04, label="rad")
    im = axes[0, 1].imshow(obj_phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("Recon phase (mean-subtracted)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, label="rad")
    clim = max(float(np.abs(phase_diff).max()), 1e-6)
    im = axes[0, 2].imshow(phase_diff, cmap="RdBu_r", vmin=-clim, vmax=clim)
    axes[0, 2].set_title(f"Phase difference\nNCC={phase_ncc:.3f} NRMSE={phase_nrmse:.3f}")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label="rad")
    axes[1, 0].imshow(_complex_to_hsv(gt))
    axes[1, 0].set_title("GT complex HSV")
    axes[1, 1].imshow(_complex_to_hsv(obj))
    axes[1, 1].set_title("Recon complex HSV")
    amp_dev = np.abs(obj) / (np.abs(obj).mean() + 1e-12) - 1.0
    clim_a = max(0.05, float(np.abs(amp_dev).max()))
    im = axes[1, 2].imshow(amp_dev, cmap="RdBu_r", vmin=-clim_a, vmax=clim_a)
    axes[1, 2].set_title("Amplitude deviation |O|/mean - 1")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"CP reconstruction vs ground truth  Phase NCC={phase_ncc:.4f}  NRMSE={phase_nrmse:.4f}", fontsize=12)
    fig.tight_layout()
    _save(fig, "06_cp_phase_comparison.png", created)


def visualize_confocal(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    meas = raw["meas"]
    wall_size = float(raw["wall_size"])
    bin_resolution = float(raw["bin_resolution"])
    vol = np.load(_output_npz(run))["fk"]
    nt, ny, nx = vol.shape
    hw = wall_size / 2.0
    z_max = nt * 3e8 * bin_resolution / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(np.log1p(meas[:, :, meas.shape[2] // 3]), cmap="gray", origin="upper")
    axes[0].set_title("Transient slice (log)")
    axes[1].imshow(np.log1p(meas.max(axis=2)), cmap="gray", origin="upper")
    axes[1].set_title("Wall max projection (log)")
    hist = meas[meas.shape[0] // 2, meas.shape[1] // 2, :]
    axes[2].plot(np.arange(hist.size) * bin_resolution * 1e9, hist, lw=0.8)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Photons")
    axes[2].set_title("Center transient")
    for ax in axes[:2]:
        ax.axis("off")
    fig.suptitle("Confocal NLOS measurements", fontsize=12)
    fig.tight_layout()
    _save(fig, "07_confocal_measurements.png", created)

    front = vol.max(axis=0)
    top = vol.max(axis=1)
    side = vol.max(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_norm(front), cmap="gray", origin="upper", extent=[-hw, hw, hw, -hw], aspect="equal")
    axes[0].set_title("Front view")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].imshow(_norm(top), cmap="gray", origin="upper", extent=[-hw, hw, 0, z_max], aspect="auto")
    axes[1].set_title("Top view")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[2].imshow(_norm(side.T), cmap="gray", origin="upper", extent=[0, z_max, hw, -hw], aspect="auto")
    axes[2].set_title("Side view")
    axes[2].set_xlabel("z (m)")
    axes[2].set_ylabel("y (m)")
    fig.suptitle("Confocal NLOS f-k reconstruction projections", fontsize=12)
    fig.tight_layout()
    _save(fig, "08_confocal_fk_three_views.png", created)

    v = _norm(vol)
    zz, yy, xx = np.where(v > 0.12)
    vals = v[zz, yy, xx]
    if vals.size > 100_000:
        idx = np.linspace(0, vals.size - 1, 100_000, dtype=int)
        zz, yy, xx, vals = zz[idx], yy[idx], xx[idx], vals[idx]
    fig = plt.figure(figsize=(12, 5), facecolor="black")
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black")
    cm = plt.get_cmap("hot")
    rgba = cm(vals ** 0.5)
    rgba[:, 3] = np.clip(vals ** 2.0, 0, 1)
    ax.scatter(np.linspace(-hw, hw, nx)[xx], np.linspace(0, z_max, nt)[zz], -np.linspace(-hw, hw, ny)[yy],
               c=rgba, s=1.2, linewidths=0, depthshade=False)
    ax.set_xlim(-hw, hw)
    ax.set_ylim(0, z_max)
    ax.set_zlim(-hw, hw)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("y (m)")
    ax.set_title("3-D f-k volume", color="white")
    ax.view_init(elev=25, azim=-50)
    ax.tick_params(colors="white", labelsize=6)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_color("white")
    ax2 = fig.add_subplot(1, 2, 2)
    z_axis = np.linspace(0, z_max, nt)
    profile = vol.max(axis=(1, 2))
    profile = profile / profile.max() if profile.max() > 0 else profile
    ax2.plot(z_axis, profile, lw=1.5)
    ax2.set_xlabel("Depth z (m)")
    ax2.set_ylabel("Normalised max intensity")
    ax2.set_title("Depth profile")
    fig.suptitle("Confocal NLOS 3-D volume and depth profile", color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, "09_confocal_3d_depth.png", created)


def visualize_lsrtm(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    v_mig = _squeeze_first(raw["v_mig"])
    observed = _squeeze_first(raw["observed_data"])
    recon = np.load(_output_npz(run))["v_true"]
    gt = _squeeze_first(np.load(_task_data(run, "data/ground_truth.npz"))["v_true"])
    meta = json.loads((ws / "data" / "meta_data.json").read_text(encoding="utf-8"))
    dx = float(meta.get("preprocessing", {}).get("dx_m", 1.0))
    extent = [0, (gt.shape[0] - 1) * dx, (gt.shape[1] - 1) * dx, 0]
    vmin, vmax = gt.min(), gt.max()

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
    for ax, arr, title in zip(axes, [gt, v_mig, recon], ["True Velocity", "Migration Velocity", "PASS Output Velocity"]):
        im = ax.imshow(arr.T, aspect="auto", cmap="viridis", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Depth (m)")
        fig.colorbar(im, ax=ax, label="m/s", shrink=0.8)
    axes[-1].set_xlabel("Distance (m)")
    fig.suptitle(f"LSRTM velocity models  NCC={_ncc(recon, gt):.4f}  NRMSE={_nrmse(recon, gt):.4f}", fontsize=12)
    fig.tight_layout()
    _save(fig, "10_lsrtm_velocity_models.png", created)

    shot_idx = 0
    obs = observed[shot_idx]
    clip = np.percentile(np.abs(obs), 98)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    axes[0].imshow(obs.T, aspect="auto", cmap="gray", vmin=-clip, vmax=clip)
    axes[0].set_title("Observed shot gather")
    diff_v = recon - gt
    clip_v = np.percentile(np.abs(diff_v), 98)
    axes[1].imshow(diff_v.T, aspect="auto", cmap="RdBu_r", vmin=-clip_v, vmax=clip_v, extent=extent)
    axes[1].set_title("Velocity error (output - true)")
    axes[2].imshow((recon - v_mig).T, aspect="auto", cmap="gray", extent=extent)
    axes[2].set_title("Output - migration velocity")
    axes[0].set_xlabel("Receiver")
    axes[0].set_ylabel("Time sample")
    for ax in axes[1:]:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    fig.suptitle("LSRTM notebook-style data and model diagnostics", fontsize=12)
    fig.tight_layout()
    _save(fig, "11_lsrtm_data_error.png", created)


def visualize_sos(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    sinogram = _squeeze_first(raw["sinogram"])
    sinogram_full = _squeeze_first(raw["sinogram_full"])
    angles = _squeeze_first(raw["angles"])
    angles_full = _squeeze_first(raw["angles_full"])
    recon = np.load(_output_npz(run))["sos_phantom"]
    gt_npz = np.load(_task_data(run, "data/ground_truth.npz"))
    sos_gt = _squeeze_first(gt_npz["sos_phantom"])
    delta_s_gt = _squeeze_first(gt_npz["slowness_perturbation"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(sinogram_full, aspect="auto", cmap="inferno", extent=[angles_full[0], angles_full[-1], sinogram_full.shape[0], 0])
    axes[0].set_title(f"Full Sinogram ({len(angles_full)} angles)")
    axes[0].set_xlabel("Angle (deg)")
    axes[0].set_ylabel("Detector index")
    axes[1].imshow(sinogram, aspect="auto", cmap="inferno", extent=[angles[0], angles[-1], sinogram.shape[0], 0])
    axes[1].set_title(f"Sparse Sinogram ({len(angles)} angles, with noise)")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Detector index")
    fig.tight_layout()
    _save(fig, "12_sos_sinograms.png", created)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    vmin, vmax = 1400, 2600
    im = axes[0, 0].imshow(sos_gt, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth SoS")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8, label="m/s")
    im = axes[0, 1].imshow(recon, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("PASS Output SoS")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8, label="m/s")
    im = axes[0, 2].imshow(np.abs(recon - sos_gt), cmap="hot", vmin=0, vmax=150)
    axes[0, 2].set_title("|Error|")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8, label="m/s")
    im = axes[1, 0].imshow(delta_s_gt * 1e6, cmap="RdBu_r")
    axes[1, 0].set_title("GT slowness perturbation (us/m)")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    row = sos_gt.shape[0] // 2
    axes[1, 1].plot(sos_gt[row, :], "k-", linewidth=2.5, label="Ground Truth")
    axes[1, 1].plot(recon[row, :], "r-.", linewidth=1.5, label="PASS Output")
    axes[1, 1].set_xlabel("Pixel")
    axes[1, 1].set_ylabel("Speed of Sound (m/s)")
    axes[1, 1].set_title(f"Center profile (row {row})")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].hist((recon - sos_gt).ravel(), bins=80, color="steelblue")
    axes[1, 2].set_title("Error histogram")
    for ax in axes[:, :1].ravel().tolist() + axes[0, 1:].ravel().tolist():
        ax.axis("off")
    fig.suptitle(f"Speed-of-Sound reconstruction  NCC={_ncc(recon, sos_gt):.4f}  NRMSE={_nrmse(recon, sos_gt):.4f}", fontsize=14)
    fig.tight_layout()
    _save(fig, "13_sos_reconstruction_comparison.png", created)


def visualize_fwi(run: PassRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    v_init = raw["v_init"]
    observed = raw["observed_data"]
    dx = float(raw["dx"])
    dt = float(raw["dt"])
    freq = float(raw["freq"])
    recon = np.load(_output_npz(run))["v_true"]
    gt = np.load(_task_data(run, "data/ground_truth.npz"))["v_true"]

    nt = observed.shape[2]
    t = np.arange(nt) * dt
    shot_idx = min(5, observed.shape[0] - 1)
    gather = observed[shot_idx]
    clip = np.percentile(np.abs(gather), 98)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].imshow(gather.T, aspect="auto", cmap="RdBu", vmin=-clip, vmax=clip, extent=[0, gather.shape[0] - 1, t[-1], 0])
    axes[0].set_xlabel("Receiver index")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title(f"Observed shot gather - shot {shot_idx}")
    fig.colorbar(im, ax=axes[0], label="Pressure")
    freqs = np.fft.rfftfreq(nt, d=dt)
    spec = np.abs(np.fft.rfft(gather[gather.shape[0] // 2]))
    axes[1].plot(freqs, spec / (spec.max() + 1e-12), "b-", lw=1.5)
    axes[1].axvline(freq, color="r", ls="--", label=f"f={freq:g} Hz")
    axes[1].set_xlim(0, 30)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalised amplitude")
    axes[1].set_title("Receiver trace spectrum")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "14_fwi_observed_gather.png", created)

    extent = [0, (gt.shape[0] - 1) * dx / 1000, (gt.shape[1] - 1) * dx / 1000, 0]
    vmin, vmax = gt.min(), gt.max()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for ax, arr, title in zip(axes, [v_init, recon, gt], ["Initial Velocity Model", "PASS Output Velocity Model", "True Velocity Model"]):
        im = ax.imshow(arr.T, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
        fig.colorbar(im, ax=ax, label="m/s", shrink=0.9)
    fig.suptitle(f"FWI velocity models  NCC={_ncc(recon, gt):.4f}  NRMSE={_nrmse(recon, gt):.4f}", fontsize=12)
    fig.tight_layout()
    _save(fig, "15_fwi_velocity_models.png", created)

    diff = recon - gt
    clip_v = np.percentile(np.abs(diff), 98)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(diff.T, aspect="auto", cmap="RdBu", vmin=-clip_v, vmax=clip_v, extent=extent)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"Velocity error (output - true), rel. L2={np.linalg.norm(diff)/np.linalg.norm(gt)*100:.2f}%")
    fig.colorbar(im, ax=ax, label="m/s")
    fig.tight_layout()
    _save(fig, "16_fwi_velocity_error.png", created)


VISUALIZERS = {
    "SSNP_ODT": visualize_ssnp,
    "conventional_ptychography": visualize_ptychography,
    "confocal-nlos-fk": visualize_confocal,
    "seismic_lsrtm_original": visualize_lsrtm,
    "ultrasound_sos_tomography": visualize_sos,
    "seismic_FWI_original": visualize_fwi,
}


def _write_index(created: Iterable[str]) -> None:
    cards = []
    for name in created:
        title = name.rsplit(".", 1)[0].replace("_", " ")
        cards.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<a href='{html.escape(name)}'><img src='{html.escape(name)}' alt='{html.escape(title)}'></a>"
            "</section>"
        )
    body = "\n".join(cards)
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wave Optics Train PASS Visualizations</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f4; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #555; }}
    section {{ margin: 28px 0; padding: 16px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Wave Optics Train PASS Visualizations</h1>
  <p>Notebook-style visualizations for the six train tasks with PASS runs.</p>
  {body}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(index, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for run in PASS_RUNS:
        summary = _load_summary(run)
        if summary.get("verdict") != "PASS":
            raise RuntimeError(f"{run.task_id}/{run.run_id} is not PASS")
        VISUALIZERS[run.task_id](run, created)
    _write_index(created)
    print(f"Wrote {len(created)} figures to {OUT_DIR}")
    print(f"Open {OUT_DIR / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
