# Hamiltonian Integration Length vs Curvature (2D Banana)

Test the idea that in HMC/NUTS, the integration length `L` (roughly `step_size * num_integration_steps`) is correlated with local posterior geometry. Use a 2D banana target for this because curvature is tunable through `b`.

Tooling:
- JAX for geometry and autodiff Hessians
- BlackJAX for NUTS

## Setup

```bash
cd /mnt/c/Users/guile/Documents/code/geometry_experiments/hamiltonian_integration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Baseline run (single banana b)

```bash
python banana_hmc_length_experiment.py --warmup 2000 --samples 30000 --seed 42 --banana-b 0.2
```

Write outputs to `outputs/`.

## Curvature sweep run

```bash
python banana_b_sweep.py --warmup 1500 --samples 18000 --b-values 0.05,0.1,0.2,0.35,0.5
```

This adds sweep artifacts in `outputs/` (heatmap panels + metrics).

## Neal's funnel run (2D)

```bash
python neals_funnel_hmc_length_experiment.py --warmup 2000 --samples 30000 --seed 42 --y-scale 3.0
```

This writes to `outputs/neals_funnel/` by default, so banana artifacts are not overwritten.

Neal's funnel artifacts:
- `neals_funnel_heatmaps_length_vs_posterior.png`
- `neals_funnel_leading_eigenvector_overlay.png`
- `neals_funnel_eigenvector_alignment_heatmap.png`
- `neals_funnel_summary.txt`

## Notebook story


```bash
jupyter notebook integration_length_curvature_story.ipynb
```

The notebook runs baseline and sweep, then displays the generated figures in sequence.

## Artifacts produced

Baseline:
- `heatmaps_length_vs_posterior.png`
- `leading_eigenvector_overlay.png`
- `eigenvector_alignment_heatmap.png`
- `summary.txt`

Sweep:
- `sweep_mean_L_heatmaps.png`
- `sweep_metrics.png`
- `sweep_metrics.csv`

## Notes

- Potential energy is `U(q) = 0.5 z(q)^T Σ^{-1} z(q)` with
  `z = [q1, q2 - b(q1^2 - a^2)]`.
- Compare an explicit analytic Hessian with a JAX autodiff Hessian pointwise.
- Sample size reasonably high to stabilize heatmaps.
