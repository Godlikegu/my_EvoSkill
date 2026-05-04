"""Notebook-style visualizations for the four wave-optics valid runs.

This is an operator-side visualization utility. It reads task reference
artifacts for side-by-side plots, but it is not part of distillation or agent
execution and does not feed valid evidence back into the skill.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT.parent / "tasks"
VALID_SLUG = "Vendor2_Claude-4.6-opus_rounda_e2e_20260504_081457_valid_20260504_152035"
LOG_ROOT = REPO_ROOT / "artifacts" / "logs" / VALID_SLUG
OUT_DIR = REPO_ROOT / "artifacts" / "visualizations" / "valid_rounda_e2e_4"


@dataclass(frozen=True)
class ValidRun:
    task_id: str
    run_id: str


VALID_RUNS = (
    ValidRun("reflection_ODT", "run-1777879235-c50f96"),
    ValidRun("xray_ptychography_tike", "run-1777881206-897135"),
    ValidRun("plane_wave_ultrasound", "run-1777882636-8c169d"),
    ValidRun("usct_FWI", "run-1777883719-1835ed"),
)


def _summary(run: ValidRun) -> dict:
    return json.loads((LOG_ROOT / run.task_id / run.run_id / "run_summary.json").read_text(encoding="utf-8"))


def _workspace(run: ValidRun) -> Path:
    return Path(_summary(run)["workspace_root"])


def _output_npz(run: ValidRun) -> Path:
    summary = _summary(run)
    rel = (
        summary.get("primary_output_path")
        or summary.get("policy", {}).get("primary_output_rel")
        or "output/reconstruction.npz"
    )
    path = Path(rel)
    return path if path.is_absolute() else Path(summary["workspace_root"]) / path


def _task_file(run: ValidRun, rel: str) -> Path:
    return TASK_ROOT / run.task_id / rel


def _save(fig: plt.Figure, name: str, created: list[str], *, dpi: int = 160) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    created.append(name)


def _squeeze(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _ncc(a: np.ndarray, b: np.ndarray, *, centered: bool = False) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if centered:
        x = x - x.mean()
        y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else 0.0


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)) / (y.max() - y.min() + 1e-12))


def _latest_judge_metrics(run: ValidRun) -> dict:
    judges = sorted((LOG_ROOT / run.task_id / run.run_id).glob("judge_round_*.json"))
    if not judges:
        return {}
    data = json.loads(judges[-1].read_text(encoding="utf-8"))
    return data.get("judge_result", {}).get("metrics_actual", {})


def visualize_reflection_odt(run: ValidRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    meas = _squeeze(raw["measurements"])
    recon = _squeeze(np.load(_output_npz(run))["delta_n"])
    gt = _squeeze(np.load(_task_file(run, "data/ground_truth.npz"))["delta_n"])
    meta = json.loads(_task_file(run, "data/meta_data.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(6, 6))
    na_obj = float(meta["NA_obj"])
    for ring in meta["illumination_rings"]:
        angles = np.linspace(0, 2 * np.pi, int(ring["n_angles"]), endpoint=False)
        color = "tab:blue" if ring["type"] == "BF" else "tab:red"
        ax.scatter(ring["NA"] * np.cos(angles), ring["NA"] * np.sin(angles), s=55, label=f"{ring['type']} NA={ring['NA']:.3f}")
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(na_obj * np.cos(theta), na_obj * np.sin(theta), "k--", lw=1.5, label=f"Objective NA={na_obj:.2f}")
    ax.set_aspect("equal")
    ax.set_xlabel("NAx")
    ax.set_ylabel("NAy")
    ax.set_title("Reflection ODT illumination angle distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "01_reflection_odt_illumination.png", created)

    fig, axes = plt.subplots(7, 8, figsize=(16, 13))
    labels = []
    for ring in meta["illumination_rings"]:
        labels += [f"{ring['type']} NA={ring['NA']:.3f}\nAngle {i}" for i in range(int(ring["n_angles"]))]
    for i, ax in enumerate(axes.ravel()):
        if i < meas.shape[0]:
            ax.imshow(meas[i], cmap="gray", origin="lower")
            ax.set_title(labels[i], fontsize=7)
        ax.axis("off")
    fig.suptitle("Simulated Reflection-Mode IDT Measurements", fontsize=12)
    fig.tight_layout()
    _save(fig, "02_reflection_odt_measurements.png", created)

    vmax = max(np.percentile(np.abs(gt), 99.5), np.percentile(np.abs(recon), 99.5))
    vmin = -vmax if min(gt.min(), recon.min()) < 0 else 0.0
    fig, axes = plt.subplots(3, gt.shape[0], figsize=(3.5 * gt.shape[0], 10))
    for iz in range(gt.shape[0]):
        axes[0, iz].imshow(gt[iz], cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, iz].set_title(f"Ground Truth (z={iz})")
        axes[1, iz].imshow(recon[iz], cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[1, iz].set_title(f"Agent Reconstruction (z={iz})")
        im = axes[2, iz].imshow(np.abs(gt[iz] - recon[iz]), cmap="hot", origin="lower")
        axes[2, iz].set_title("|Difference|")
        fig.colorbar(im, ax=axes[2, iz], fraction=0.046, pad=0.04)
        for r in range(3):
            axes[r, iz].axis("off")
    m = _latest_judge_metrics(run)
    fig.suptitle(
        f"Reflection-Mode ODT comparison  NCC={m.get('ncc', _ncc(recon, gt, centered=True)):.4f}  NRMSE={m.get('nrmse', _nrmse(recon, gt)):.4f}",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "03_reflection_odt_comparison.png", created)


def visualize_xray_ptychography(run: ValidRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    patterns = _squeeze(raw["diffraction_patterns"])
    scan = _squeeze(raw["scan_positions"])
    probe = _squeeze(raw["probe_guess"])
    est = _squeeze(np.load(_output_npz(run))["object_phase"])
    ref_npz = np.load(_task_file(run, "data/baseline_reference.npz"))
    ref_phase = _squeeze(ref_npz["object_phase"])
    ref_amp = _squeeze(ref_npz["object_amplitude"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(scan[:, 1], scan[:, 0], s=3, alpha=0.6)
    ax.set_xlabel("Position Y (pixels)")
    ax.set_ylabel("Position X (pixels)")
    ax.set_title(f"Scan Positions ({scan.shape[0]} points)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_xray_ptycho_scan_positions.png", created)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx in zip(axes, [0, patterns.shape[0] // 2, patterns.shape[0] - 1]):
        im = ax.imshow(np.log1p(patterns[idx]), cmap="viridis")
        ax.set_title(f"Pattern #{idx} (log scale)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Example Diffraction Patterns", fontsize=14)
    fig.tight_layout()
    _save(fig, "05_xray_ptycho_diffraction_patterns.png", created)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(np.abs(probe), cmap="hot")
    axes[0].set_title("Initial Probe Amplitude")
    fig.colorbar(im, ax=axes[0])
    im = axes[1].imshow(np.angle(probe), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Initial Probe Phase")
    fig.colorbar(im, ax=axes[1])
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Initial Probe Guess", fontsize=14)
    fig.tight_layout()
    _save(fig, "06_xray_ptycho_probe_guess.png", created)

    h = min(ref_phase.shape[0], est.shape[0])
    w = min(ref_phase.shape[1], est.shape[1])
    ref = ref_phase[:h, :w]
    est = est[:h, :w]
    diff = est - ref
    lim = max(float(np.percentile(np.abs(diff), 99)), 1e-6)
    m = _latest_judge_metrics(run)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    im = axes[0, 0].imshow(ref, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title("Baseline Reference Phase")
    fig.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(est, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("Agent Reconstructed Phase")
    fig.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0, 2].set_title("Phase Difference")
    fig.colorbar(im, ax=axes[0, 2])
    im = axes[1, 0].imshow(ref_amp[:h, :w], cmap="gray")
    axes[1, 0].set_title("Baseline Reference Amplitude")
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 1].hist(diff.ravel(), bins=100, color="steelblue")
    axes[1, 1].set_title("Phase difference histogram")
    row = h // 2
    axes[1, 2].plot(ref[row], "k-", lw=1.5, label="Reference")
    axes[1, 2].plot(est[row], "r-", lw=1.0, label="Agent")
    axes[1, 2].set_title(f"Center phase profile (row {row})")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    for ax in axes.ravel()[:4]:
        ax.axis("off")
    fig.suptitle(
        f"Phase Comparison  NCC={m.get('ncc', _ncc(est, ref)):.4f}  NRMSE={m.get('nrmse', _nrmse(est, ref)):.4f}",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, "07_xray_ptycho_phase_comparison.png", created)


def _bmode_axes(task: str, name: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ref_dir = TASK_ROOT / task / "evaluation" / "reference_outputs"
    x_path = ref_dir / f"x_{name}.npy"
    z_path = ref_dir / f"z_{name}.npy"
    if x_path.exists() and z_path.exists():
        return np.load(x_path), np.load(z_path)
    x = np.arange(shape[1], dtype=float)
    z = np.arange(shape[0], dtype=float)
    return x, z


def _plot_us(img: np.ndarray, x: np.ndarray, z: np.ndarray, ax, title: str) -> None:
    data = np.asarray(img, dtype=float)
    data = data / (np.nanmax(data) + 1e-12)
    dx = np.mean(np.diff(x)) if x.size > 1 else 1.0
    dz = np.mean(np.diff(z)) if z.size > 1 else 1.0
    extent = [x.min() - dx, x.max() + dx, z.max() + dz, z.min() - dz]
    ax.imshow(data, cmap="gray", extent=extent, interpolation="none", aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("Azimuth (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)


def visualize_plane_wave_ultrasound(run: ValidRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    out = np.load(_output_npz(run))
    ref = np.load(_task_file(run, "data/baseline_reference.npz"))
    b_fib = _squeeze(out["bmode_fibers"])
    b_cys = _squeeze(out["bmode_cysts"])
    r_fib = _squeeze(ref["bmode_fibers"])
    r_cys = _squeeze(ref["bmode_cysts"])
    x_fib, z_fib = _bmode_axes(run.task_id, "fibers", b_fib.shape)
    x_cys, z_cys = _bmode_axes(run.task_id, "cysts", b_cys.shape)
    rf_fib = _squeeze(raw["RF_fibers"])
    rf_cys = _squeeze(raw["RF_cysts"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    _plot_us(np.abs(rf_fib[:, :, rf_fib.shape[-1] // 2]), x_fib, z_fib, axes[0], "Raw RF data (center angle)\nbefore migration")
    _plot_us(b_fib, x_fib, z_fib, axes[1], "Agent B-mode fibers\nafter migration")
    fig.tight_layout()
    _save(fig, "08_plane_wave_before_after_fibers.png", created)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    _plot_us(r_fib, x_fib, z_fib, axes[0], "Reference fibers B-mode")
    _plot_us(b_fib, x_fib, z_fib, axes[1], "Agent fibers B-mode")
    _plot_us(np.abs(b_fib / (b_fib.max() + 1e-12) - r_fib / (r_fib.max() + 1e-12)), x_fib, z_fib, axes[2], "|Normalized difference|")
    m = _latest_judge_metrics(run)
    fig.suptitle(
        f"Wire-target phantom  NCC={m.get('ncc', _ncc(b_fib, r_fib)):.4f}  PSF FWHM={m.get('psf_fwhm_mm_mean', float('nan')):.2f} mm",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "09_plane_wave_fibers_comparison.png", created)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    _plot_us(r_cys, x_cys, z_cys, axes[0], "Reference cyst B-mode")
    _plot_us(b_cys, x_cys, z_cys, axes[1], "Agent cyst B-mode")
    _plot_us(np.abs(b_cys / (b_cys.max() + 1e-12) - r_cys / (r_cys.max() + 1e-12)), x_cys, z_cys, axes[2], "|Normalized difference|")
    fig.suptitle("Circular cyst phantom B-mode comparison", fontsize=13)
    fig.tight_layout()
    _save(fig, "10_plane_wave_cysts_comparison.png", created)

    z_idx = min(len(z_cys) - 1, int(np.argmin(np.abs(z_cys - 0.021))))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_us(b_cys, x_cys, z_cys, axes[0], "Agent cyst phantom with profile depth")
    axes[0].axhline(z_cys[z_idx], color="tab:red", lw=1)
    axes[1].plot(x_cys * 1e3, r_cys[z_idx] / (r_cys[z_idx].max() + 1e-12), "k-", label="Reference")
    axes[1].plot(x_cys * 1e3, b_cys[z_idx] / (b_cys[z_idx].max() + 1e-12), "b-", label="Agent")
    axes[1].set_xlabel("Azimuth (mm)")
    axes[1].set_ylabel("Normalized B-mode")
    axes[1].set_title(f"Lateral profile at z = {z_cys[z_idx] * 1e3:.1f} mm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "11_plane_wave_cyst_profile.png", created)


def visualize_usct_fwi(run: ValidRun, created: list[str]) -> None:
    ws = _workspace(run)
    raw = np.load(ws / "data" / "raw_data.npz")
    recon = _squeeze(np.load(_output_npz(run))["vp_reconstructed"])
    ref = _squeeze(np.load(_task_file(run, "data/baseline_reference.npz"))["vp_reconstructed"])
    meta = json.loads(_task_file(run, "data/meta_data.json").read_text(encoding="utf-8"))
    domain = meta.get("domain_size_cm", [24, 24])
    extent = [0, domain[0], 0, domain[1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, freq in zip(axes, [0.3, 0.75, 1.25]):
        key = f"dobs_{freq:g}"
        im = ax.imshow(np.abs(_squeeze(raw[key])), origin="lower")
        ax.set_title(f"|dobs| at {freq:g} MHz")
        ax.set_xlabel("Source index")
        ax.set_ylabel("Receiver index")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, "12_usct_fwi_observed_data.png", created)

    ix = _squeeze(raw["receiver_ix"])
    iy = _squeeze(raw["receiver_iy"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ix, iy, s=8, alpha=0.7)
    ax.set_title(f"{len(ix)} Transducers (circular ring array)")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "13_usct_fwi_transducer_geometry.png", created)

    vmin = min(ref.min(), recon.min())
    vmax = max(ref.max(), recon.max())
    diff = recon - ref
    vd = max(float(np.percentile(np.abs(diff), 99)), 1e-6)
    m = _latest_judge_metrics(run)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im = axes[0].imshow(ref, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("Reference")
    fig.colorbar(im, ax=axes[0], label="Sound speed (m/s)")
    im = axes[1].imshow(recon, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title(f"Agent reconstruction (NCC={m.get('ncc', _ncc(recon, ref)):.4f})")
    fig.colorbar(im, ax=axes[1], label="Sound speed (m/s)")
    im = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd, extent=extent)
    axes[2].set_title(f"Difference (max |err|={vd:.1f} m/s)")
    fig.colorbar(im, ax=axes[2], label="Delta c (m/s)")
    for ax in axes:
        ax.set_xlabel("X / cm")
        ax.set_ylabel("Y / cm")
    fig.suptitle(
        f"USCT FWI comparison  NCC={m.get('ncc', _ncc(recon, ref)):.4f}  NRMSE={m.get('nrmse', _nrmse(recon, ref)):.4f}",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "14_usct_fwi_comparison.png", created)

    row = recon.shape[0] // 2
    col = recon.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ref[row], "k-", label="Reference")
    axes[0].plot(recon[row], "r-", label="Agent")
    axes[0].set_title(f"Center row profile (row {row})")
    axes[1].plot(ref[:, col], "k-", label="Reference")
    axes[1].plot(recon[:, col], "r-", label="Agent")
    axes[1].set_title(f"Center column profile (col {col})")
    for ax in axes:
        ax.set_ylabel("Sound speed (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    _save(fig, "15_usct_fwi_profiles.png", created)


VISUALIZERS = {
    "reflection_ODT": visualize_reflection_odt,
    "xray_ptychography_tike": visualize_xray_ptychography,
    "plane_wave_ultrasound": visualize_plane_wave_ultrasound,
    "usct_FWI": visualize_usct_fwi,
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
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wave Optics Valid Skill-Only Visualizations</title>
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
  <h1>Wave Optics Valid Skill-Only Visualizations</h1>
  <p>Notebook-style plots for the four valid tasks from slug <code>{html.escape(VALID_SLUG)}</code>.</p>
  {''.join(cards)}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(index, encoding="utf-8")


def main() -> int:
    created: list[str] = []
    for run in VALID_RUNS:
        VISUALIZERS[run.task_id](run, created)
    _write_index(created)
    print(f"Wrote {len(created)} figures to {OUT_DIR}")
    print(f"Open {OUT_DIR / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
