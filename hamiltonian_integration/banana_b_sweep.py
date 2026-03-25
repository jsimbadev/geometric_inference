#!/usr/bin/env python3
"""Sweep banana curvature parameter b and compare realized HMC integration-length signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from banana_hmc_length_experiment import binned_stat2d, run_nuts_chain


def parse_b_values(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No b values provided")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep banana b and compare local HMC integration length")
    parser.add_argument("--b-values", type=str, default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--banana-a", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=18000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    b_values = parse_b_values(args.b_values)
    runs = []

    for i, b in enumerate(b_values):
        qs, ls, accepts, step_size = run_nuts_chain(
            num_warmup=args.warmup,
            num_samples=args.samples,
            seed=args.seed + i,
            a=args.banana_a,
            b=b,
        )
        runs.append({"b": b, "q": qs, "L": ls, "accepts": accepts, "step_size": step_size})

    all_q = np.vstack([r["q"] for r in runs])
    lo1, hi1 = np.quantile(all_q[:, 0], [0.01, 0.99])
    lo2, hi2 = np.quantile(all_q[:, 1], [0.01, 0.99])
    xedges = np.linspace(lo1, hi1, args.bins + 1)
    yedges = np.linspace(lo2, hi2, args.bins + 1)

    n = len(runs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.7 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx, run in enumerate(runs):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        q1, q2 = run["q"][:, 0], run["q"][:, 1]
        mean_l, counts = binned_stat2d(q1, q2, run["L"], xedges, yedges, mode="mean")
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(mean_l.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(f"b={run['b']:.3f}")
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        # hatch sparse bins for visibility
        sparse = np.where(counts.T < 4, 1.0, np.nan)
        ax.contourf(
            np.linspace(extent[0], extent[1], sparse.shape[1]),
            np.linspace(extent[2], extent[3], sparse.shape[0]),
            sparse,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    cbar.set_label("Mean L")
    fig.savefig(args.outdir / "sweep_mean_L_heatmaps.png", dpi=180)
    plt.close(fig)

    rows_out = []
    for run in runs:
        ls = run["L"]
        rows_out.append(
            (
                run["b"],
                float(run["step_size"]),
                float(np.mean(ls)),
                float(np.std(ls)),
                float(np.nanmean(run["accepts"])),
            )
        )

    metrics = np.array(rows_out, dtype=np.float64)
    header = "b,step_size,mean_L,std_L,mean_acceptance"
    np.savetxt(args.outdir / "sweep_metrics.csv", metrics, delimiter=",", header=header, comments="")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.plot(metrics[:, 0], metrics[:, 2], marker="o", label="mean L")
    ax.plot(metrics[:, 0], metrics[:, 3], marker="s", label="std L")
    ax.set_xlabel("banana b")
    ax.set_ylabel("length")
    ax.set_title("Integration length moments vs b")
    ax.legend()

    fig.savefig(args.outdir / "sweep_metrics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
