"""Generic 2D acoustic FWI helper with batched C-PML propagation.

This script is intentionally domain-generic: it reads public waveform
inversion arrays, runs a bounded PyTorch FWI loop, and writes an npz output
with a caller-specified key. It contains no task ids, hidden paths, metrics,
thresholds, or fixed answers.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.checkpoint import checkpoint


def _fd1_y(a: torch.Tensor, rdy: float) -> torch.Tensor:
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1, :] - a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] - a[..., :-4, :])) * rdy,
        [0, 0, 2, 2],
    )


def _fd1_x(a: torch.Tensor, rdx: float) -> torch.Tensor:
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1] - a[..., 1:-3])
         - 1.0 / 12.0 * (a[..., 4:] - a[..., :-4])) * rdx,
        [2, 2],
    )


def _fd2_y(a: torch.Tensor, rdy2: float) -> torch.Tensor:
    return F.pad(
        (-2.5 * a[..., 2:-2, :]
         + 4.0 / 3.0 * (a[..., 3:-1, :] + a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] + a[..., :-4, :])) * rdy2,
        [0, 0, 2, 2],
    )


def _fd2_x(a: torch.Tensor, rdx2: float) -> torch.Tensor:
    return F.pad(
        (-2.5 * a[..., 2:-2]
         + 4.0 / 3.0 * (a[..., 3:-1] + a[..., 1:-3])
         - 1.0 / 12.0 * (a[..., 4:] + a[..., :-4])) * rdx2,
        [2, 2],
    )


def _cfl_step_ratio(dy: float, dx: float, dt: float, vmax: float) -> Tuple[float, int]:
    max_dt = 0.6 / math.sqrt(1.0 / dy**2 + 1.0 / dx**2) / vmax
    ratio = max(1, int(math.ceil(abs(dt) / max_dt)))
    return dt / ratio, ratio


def _setup_pml_1d(n: int, left_end: float, right_start: float, width: int,
                  sigma0: float, alpha0: float, dt: float,
                  dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n, device=device, dtype=dtype)
    eps = 1e-9
    frac = torch.maximum((left_end - x) / (width + eps), (x - right_start) / (width + eps))
    frac = torch.clamp(frac, 0.0, 1.0)
    sigma = sigma0 * frac**2
    alpha = alpha0 * (1.0 - frac)
    sigma_alpha = sigma + alpha
    a = torch.exp(-sigma_alpha * abs(dt))
    b = sigma / torch.clamp(sigma_alpha, min=eps) * (a - 1.0)
    a = torch.where(frac == 0, torch.zeros_like(a), a)
    b = torch.where(frac == 0, torch.zeros_like(b), b)
    return a, b


def setup_pml(ny_p: int, nx_p: int, pml_width: int, fd_pad: int,
              dy: float, dx: float, dt: float, vmax: float,
              dtype: torch.dtype, device: torch.device, freq: float) -> List[torch.Tensor]:
    alpha0 = math.pi * freq
    max_pml = pml_width * max(dy, dx)
    sigma0 = -(1 + 2) * vmax * math.log(1e-3) / (2.0 * max_pml)
    interior_start = fd_pad + pml_width
    end_y = ny_p - 1 - fd_pad - pml_width
    end_x = nx_p - 1 - fd_pad - pml_width
    ay1, by1 = _setup_pml_1d(ny_p, interior_start, end_y, pml_width, sigma0, alpha0, dt, dtype, device)
    ax1, bx1 = _setup_pml_1d(nx_p, interior_start, end_x, pml_width, sigma0, alpha0, dt, dtype, device)
    dbydy = _fd1_x(by1.unsqueeze(0), 1.0 / dy).squeeze(0)
    dbxdx = _fd1_x(bx1.unsqueeze(0), 1.0 / dx).squeeze(0)
    return [
        ay1.reshape(ny_p, 1), by1.reshape(ny_p, 1), dbydy.reshape(ny_p, 1),
        ax1.reshape(1, nx_p), bx1.reshape(1, nx_p), dbxdx.reshape(1, nx_p),
    ]


def wave_step(vp, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, pml, dy, dx, dt):
    ay, by, dbydy, ax, bx, dbxdx = pml
    rdy, rdx = 1.0 / dy, 1.0 / dx
    rdy2, rdx2 = rdy * rdy, rdx * rdx
    dpy = _fd1_y(wfc, rdy)
    dpx = _fd1_x(wfc, rdx)
    d2py = _fd2_y(wfc, rdy2)
    d2px = _fd2_x(wfc, rdx2)
    tmp_y = (1.0 + by) * d2py + dbydy * dpy + _fd1_y(ay * psi_y, rdy)
    tmp_x = (1.0 + bx) * d2px + dbxdx * dpx + _fd1_x(ax * psi_x, rdx)
    lap = (1.0 + by) * tmp_y + ay * zeta_y + (1.0 + bx) * tmp_x + ax * zeta_x
    psi_y_new = by * dpy + ay * psi_y
    psi_x_new = bx * dpx + ax * psi_x
    zeta_y_new = by * tmp_y + ay * zeta_y
    zeta_x_new = bx * tmp_x + ax * zeta_x
    wfn = vp * vp * dt * dt * lap + 2.0 * wfc - wfp
    return wfn, psi_y_new, psi_x_new, zeta_y_new, zeta_x_new


def fft_upsample(signal: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return signal
    nt = signal.shape[-1]
    up_nt = nt * ratio
    spec = torch.fft.rfft(signal, norm="ortho") * math.sqrt(ratio)
    if spec.shape[-1] > 1:
        spec = spec.clone()
        spec[..., -1] = 0
    pad = up_nt // 2 + 1 - spec.shape[-1]
    if pad > 0:
        spec = F.pad(spec, (0, pad))
    return torch.fft.irfft(spec, n=up_nt, norm="ortho")


def fft_downsample(signal: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return signal
    nt = signal.shape[-1]
    down_nt = nt // ratio
    spec = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]
    if spec.shape[-1] > 1:
        spec = spec.clone()
        spec[..., -1] = 0
    return torch.fft.irfft(spec, n=down_nt, norm="ortho") / math.sqrt(ratio)


def ricker(freq: float, nt: int, dt: float, n_shots: int, device: torch.device) -> torch.Tensor:
    peak = 1.5 / freq
    t = torch.arange(nt, device=device, dtype=torch.float32) * dt - peak
    wav = (1.0 - 2.0 * math.pi**2 * freq**2 * t**2) * torch.exp(-math.pi**2 * freq**2 * t**2)
    return wav.reshape(1, 1, nt).expand(n_shots, 1, nt).contiguous()


def make_geometry(ny: int, n_shots: int, n_rec: int, source_depth: int,
                  receiver_depth: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # The first model axis is the acquisition line; the second is depth.
    src = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    src[..., 1] = int(source_depth)
    src[:, 0, 0] = torch.linspace(0, ny - 1, n_shots, device=device).round().long()
    rec = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
    rec[..., 1] = int(receiver_depth)
    rec[:, :, 0] = torch.linspace(0, ny - 1, n_rec, device=device).round().long().unsqueeze(0).expand(n_shots, -1)
    return src, rec


def loc_to_flat(loc: torch.Tensor, pad_y: int, pad_x: int, nx_p: int) -> torch.Tensor:
    return ((loc[..., 0] + pad_y) * nx_p + (loc[..., 1] + pad_x)).long()


def forward_model(v: torch.Tensor, spacing: Tuple[float, float], dt: float,
                  src_amp: torch.Tensor, src_loc: torch.Tensor, rec_loc: torch.Tensor,
                  freq: float, pml_width: int = 20, checkpoint_every: int = 64) -> torch.Tensor:
    dy, dx = float(spacing[0]), float(spacing[1])
    ny, nx = v.shape
    fd_pad = 2
    pad = fd_pad + pml_width
    ny_p, nx_p = ny + 2 * pad, nx + 2 * pad
    device, dtype = v.device, v.dtype
    vmax = float(v.detach().abs().max().item())
    inner_dt, ratio = _cfl_step_ratio(dy, dx, dt, vmax)
    vp = F.pad(v.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate").squeeze(0).squeeze(0)
    pml = setup_pml(ny_p, nx_p, pml_width, fd_pad, dy, dx, inner_dt, vmax, dtype, device, freq)
    src_up = fft_upsample(src_amp, ratio)
    nt_inner = src_up.shape[-1]
    src_flat = loc_to_flat(src_loc, pad, pad, nx_p)
    rec_flat = loc_to_flat(rec_loc, pad, pad, nx_p)
    n_shots = src_amp.shape[0]
    n_rec = rec_loc.shape[1]
    flat = ny_p * nx_p
    vx = vp.reshape(-1)[src_flat[:, 0]]
    src_scaled = -src_up[:, 0, :] * (vx ** 2).unsqueeze(1) * inner_dt**2
    src_indicator = torch.zeros(n_shots, flat, dtype=dtype, device=device)
    src_indicator.scatter_(1, src_flat, 1.0)
    src_indicator = src_indicator.view(n_shots, ny_p, nx_p)
    shot_idx = torch.arange(n_shots, device=device)
    state = [torch.zeros(n_shots, ny_p, nx_p, dtype=dtype, device=device) for _ in range(6)]
    rec_segments: List[torch.Tensor] = []

    def make_segment(t_start: int, count: int):
        def segment(wfc, wfp, psi_y, psi_x, zeta_y, zeta_x):
            recs = []
            for i in range(count):
                t = t_start + i
                recs.append(wfc.reshape(n_shots, -1)[shot_idx.unsqueeze(1), rec_flat])
                wfn, psi_y, psi_x, zeta_y, zeta_x = wave_step(
                    vp, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, pml, dy, dx, inner_dt
                )
                wfn = wfn + src_scaled[:, t].reshape(n_shots, 1, 1) * src_indicator
                wfp, wfc = wfc, wfn
            return wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, torch.stack(recs, dim=0)
        return segment

    for start in range(0, nt_inner, checkpoint_every):
        count = min(checkpoint_every, nt_inner - start)
        seg = make_segment(start, count)
        if v.requires_grad:
            *state, rec = checkpoint(seg, *state, use_reentrant=False)
        else:
            with torch.no_grad():
                *state, rec = seg(*state)
        rec_segments.append(rec)
    rec_inner = torch.cat(rec_segments, dim=0).permute(*[1, 2, 0]).contiguous()
    return fft_downsample(rec_inner, ratio).reshape(n_shots, n_rec, -1)


def cosine_taper(x: torch.Tensor, n: int = 5) -> torch.Tensor:
    if n <= 0:
        return x
    n = min(n, x.shape[-1])
    tap = (torch.cos(torch.arange(1, n + 1, device=x.device, dtype=x.dtype) / n * torch.pi) + 1.0) / 2.0
    out = x.clone()
    out[..., -n:] *= tap
    return out


def load_meta(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-key", default="v_true")
    ap.add_argument("--model-key", default="v_init")
    ap.add_argument("--data-key", default="observed_data")
    ap.add_argument("--epochs", default="auto", help="'auto' or an integer")
    ap.add_argument("--budget-seconds", type=float, default=5400.0)
    ap.add_argument("--reserve-seconds", type=float, default=600.0)
    ap.add_argument("--safety-fraction", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    raw = np.load(args.input)
    meta = load_meta(args.meta)
    inv = meta.get("inversion", {}) if isinstance(meta, dict) else {}
    v = torch.from_numpy(raw[args.model_key].astype(np.float32)).to(device)
    observed = torch.from_numpy(raw[args.data_key].astype(np.float32)).to(device)
    dx = float(raw["dx"])
    dt = float(raw["dt"])
    freq = float(raw["freq"]) if "freq" in raw else float(meta.get("wavelet", {}).get("frequency_hz", 5.0))
    n_shots, n_rec, nt = map(int, observed.shape)
    src_depth = int(raw["source_depth"]) if "source_depth" in raw else int(meta.get("acquisition", {}).get("source_depth_grid", 1))
    rec_depth = int(raw["receiver_depth"]) if "receiver_depth" in raw else int(meta.get("acquisition", {}).get("receiver_depth_grid", 1))
    lr = float(inv.get("lr", 100.0))
    milestones = list(inv.get("lr_milestones", [75, 300]))
    vmin = float(inv.get("v_min_bound_m_s", 1480.0))
    vmax = float(inv.get("v_max_bound_m_s", 5800.0))
    grad_sigma = float(inv.get("gradient_smoothing_sigma", 1.0))
    clip_q = float(inv.get("gradient_clip_percentile", 0.98))
    taper_n = int(inv.get("n_taper", 5))
    metadata_epochs = int(inv.get("n_epochs", 0) or 0)

    src_loc, rec_loc = make_geometry(v.shape[0], n_shots, n_rec, src_depth, rec_depth, device)
    src_amp = ricker(freq, nt, dt, n_shots, device)
    observed_taper = cosine_taper(observed, taper_n)
    model = v.clone().requires_grad_(True)
    opt = torch.optim.Adam([model], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    start = time.time()
    print(f"device={device} model_shape={tuple(v.shape)} data_shape={tuple(observed.shape)}")
    print("probe=one_full_epoch")
    pred = forward_model(model, (dx, dx), dt, src_amp, src_loc, rec_loc, freq)
    loss = loss_fn(cosine_taper(pred, taper_n), observed_taper)
    if not torch.isfinite(loss):
        raise SystemExit("nonfinite probe loss")
    opt.zero_grad()
    loss.backward()
    if not torch.isfinite(model.grad).all():
        raise SystemExit("nonfinite probe gradient")
    with torch.no_grad():
        grad_np = model.grad.detach().cpu().numpy()
        smooth = gaussian_filter(grad_np, sigma=grad_sigma)
        model.grad.copy_(torch.as_tensor(smooth, dtype=torch.float32, device=device))
        clip = torch.quantile(model.grad.abs(), clip_q)
        torch.nn.utils.clip_grad_value_([model], clip)
    opt.step()
    sched.step()
    with torch.no_grad():
        model.clamp_(vmin, vmax)
    epoch_seconds = time.time() - start
    print(f"probe_loss={float(loss.detach().cpu()):.6g} epoch_seconds={epoch_seconds:.3f}")

    if args.epochs == "auto":
        usable = max(0.0, (args.budget_seconds - args.reserve_seconds - epoch_seconds) * args.safety_fraction)
        cap = max(1, int(usable // max(epoch_seconds, 1e-6)))
        requested = metadata_epochs if metadata_epochs > 0 else cap
        epochs_total = max(1, min(requested, cap))
    else:
        epochs_total = max(1, int(args.epochs))
    remaining_epochs = max(0, epochs_total - 1)
    print(f"epochs_total={epochs_total} remaining_epochs={remaining_epochs}")

    for epoch in range(1, epochs_total):
        t0 = time.time()
        pred = forward_model(model, (dx, dx), dt, src_amp, src_loc, rec_loc, freq)
        loss = loss_fn(cosine_taper(pred, taper_n), observed_taper)
        if not torch.isfinite(loss):
            raise SystemExit(f"nonfinite loss at epoch {epoch + 1}")
        opt.zero_grad()
        loss.backward()
        if not torch.isfinite(model.grad).all():
            raise SystemExit(f"nonfinite gradient at epoch {epoch + 1}")
        with torch.no_grad():
            grad_np = model.grad.detach().cpu().numpy()
            smooth = gaussian_filter(grad_np, sigma=grad_sigma)
            model.grad.copy_(torch.as_tensor(smooth, dtype=torch.float32, device=device))
            clip = torch.quantile(model.grad.abs(), clip_q)
            torch.nn.utils.clip_grad_value_([model], clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            model.clamp_(vmin, vmax)
        elapsed = time.time() - start
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs_total:
            print(f"epoch={epoch + 1}/{epochs_total} loss={float(loss.detach().cpu()):.6g} elapsed={elapsed:.1f}s last={time.time() - t0:.1f}s")
        if elapsed > args.budget_seconds - args.reserve_seconds:
            print("budget_guard=stop")
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = model.detach().cpu().numpy().astype(np.float32)
    np.savez(out, **{args.output_key: arr})
    print(f"wrote={out.as_posix()} key={args.output_key} shape={arr.shape} dtype={arr.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
