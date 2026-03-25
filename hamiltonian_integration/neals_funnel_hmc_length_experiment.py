#!/usr/bin/env python3
"""Probe integration-length/geometry relationship for a 2D Neal's funnel target."""

from __future__ import annotations

import argparse
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from banana_hmc_length_experiment import binned_stat2d, leading_eigvec


def potential_energy(q: jnp.ndarray, y_scale: float = 3.0) -> jnp.ndarray:
    """Neal's funnel potential (up to constant):
    y ~ N(0, y_scale^2), x | y ~ N(0, exp(y)).
    """
    x, y = q[0], q[1]
    return 0.5 * (y / y_scale) ** 2 + 0.5 * x**2 * jnp.exp(-y) + 0.5 * y


def analytic_hessian(q: np.ndarray, y_scale: float = 3.0) -> np.ndarray:
    x, y = float(q[0]), float(q[1])
    ey = np.exp(-y)
    h_xx = ey
    h_xy = -x * ey
    h_yy = 1.0 / (y_scale**2) + 0.5 * x * x * ey
    return np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)


def extract_num_integration_steps(info) -> int:
    for name in ("num_integration_steps", "integration_steps", "num_steps"):
        if hasattr(info, name):
            return int(getattr(info, name))
    if isinstance(info, dict):
        for name in ("num_integration_steps", "integration_steps", "num_steps"):
            if name in info:
                return int(info[name])
    return 1


def run_nuts_chain(num_warmup: int, num_samples: int, seed: int, y_scale: float = 3.0):
    key = jax.random.PRNGKey(seed)
    initial_position = jnp.array([0.0, 0.0])
    ld = lambda q: -potential_energy(q, y_scale=y_scale)

    warmup = blackjax.window_adaptation(blackjax.nuts, ld)
    key, warmup_key = jax.random.split(key)
    (state, params), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)

    nuts = blackjax.nuts(ld, **params)
    step_size = float(params["step_size"])

    qs = np.empty((num_samples, 2), dtype=np.float64)
    ls = np.empty(num_samples, dtype=np.float64)
    accepts = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        key, sample_key = jax.random.split(key)
        state, info = nuts.step(sample_key, state)
        n_steps = extract_num_integration_steps(info)
        qs[i] = np.asarray(state.position)
        ls[i] = step_size * n_steps
        accepts[i] = float(getattr(info, "acceptance_rate", np.nan))

    return qs, ls, accepts, step_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Neal funnel HMC integration-length vs curvature experiment")
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--grid", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--y-scale", type=float, default=3.0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/neals_funnel"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    qs, ls, accepts, step_size = run_nuts_chain(args.warmup, args.samples, args.seed, y_scale=args.y_scale)
    q1, q2 = qs[:, 0], qs[:, 1]

    lo1, hi1 = np.quantile(q1, [0.01, 0.99])
    lo2, hi2 = np.quantile(q2, [0.01, 0.99])
    xedges = np.linspace(lo1, hi1, args.bins + 1)
    yedges = np.linspace(lo2, hi2, args.bins + 1)

    density, _, _ = np.histogram2d(q1, q2, bins=[xedges, yedges], density=True)
    mean_l, _ = binned_stat2d(q1, q2, ls, xedges, yedges, mode="mean")
    median_l, _ = binned_stat2d(q1, q2, ls, xedges, yedges, mode="median")
    std_l, _ = binned_stat2d(q1, q2, ls, xedges, yedges, mode="std")

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    im0 = axes[0, 0].imshow(density.T, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[0, 0].set_title("Posterior density (Neal funnel)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(mean_l.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Mean integration length L")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(median_l.T, origin="lower", extent=extent, aspect="auto", cmap="plasma")
    axes[1, 0].set_title("Median integration length L")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(std_l.T, origin="lower", extent=extent, aspect="auto", cmap="cividis")
    axes[1, 1].set_title("Std integration length L")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")

    fig.savefig(args.outdir / "neals_funnel_heatmaps_length_vs_posterior.png", dpi=180)
    plt.close(fig)

    gx = np.linspace(lo1, hi1, args.grid)
    gy = np.linspace(lo2, hi2, args.grid)
    X, Y = np.meshgrid(gx, gy, indexing="xy")

    Vx_auto = np.zeros_like(X)
    Vy_auto = np.zeros_like(Y)
    Vx_ana = np.zeros_like(X)
    Vy_ana = np.zeros_like(Y)
    cos_sim = np.zeros_like(X)

    hess_auto = jax.hessian(lambda q: potential_energy(q, y_scale=args.y_scale))

    for i in range(args.grid):
        for j in range(args.grid):
            q = np.array([X[j, i], Y[j, i]])
            h_auto = np.asarray(hess_auto(jnp.array(q)))
            h_ana = analytic_hessian(q, y_scale=args.y_scale)
            v_auto, _ = leading_eigvec(h_auto)
            v_ana, _ = leading_eigvec(h_ana)
            Vx_auto[j, i], Vy_auto[j, i] = v_auto
            Vx_ana[j, i], Vy_ana[j, i] = v_ana
            cos_sim[j, i] = abs(float(np.dot(v_auto, v_ana)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    bg = np.histogram2d(q1, q2, bins=[xedges, yedges], density=True)[0]

    for ax, vx, vy, ttl in [
        (axes[0], Vx_auto, Vy_auto, "Leading eigenvector (JAX autodiff Hessian)"),
        (axes[1], Vx_ana, Vy_ana, "Leading eigenvector (analytic Hessian)"),
    ]:
        ax.imshow(bg.T, origin="lower", extent=extent, aspect="auto", cmap="Greys", alpha=0.55)
        ax.quiver(X, Y, vx, vy, color="tab:blue", scale=30, width=0.003)
        ax.set_title(ttl)
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")

    fig.savefig(args.outdir / "neals_funnel_leading_eigenvector_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        cos_sim,
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("|cos(angle)| between autodiff/analytic leading eigenvectors")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(args.outdir / "neals_funnel_eigenvector_alignment_heatmap.png", dpi=180)
    plt.close(fig)

    summary = {
        "samples": args.samples,
        "warmup": args.warmup,
        "seed": args.seed,
        "y_scale": args.y_scale,
        "step_size": step_size,
        "mean_L": float(np.mean(ls)),
        "std_L": float(np.std(ls)),
        "mean_acceptance_rate": float(np.nanmean(accepts)),
    }
    lines = [f"{k}: {v}" for k, v in summary.items()]
    (args.outdir / "neals_funnel_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
