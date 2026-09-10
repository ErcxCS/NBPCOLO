# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **cooperative localization (COLO)** in wireless sensor networks: estimating node
positions from noisy RSS-derived pairwise range measurements, using **Nonparametric Belief
Propagation (NBP)** with bounded-box priors, a classic-MDS baseline, and a CRLB floor. Reference
papers are in `Sources/`.

`NBP/` is the legacy prototype, kept read-only for reference. `colo_project/` is the rewrite, and
the NBP port is complete. `colo_project/gnn/` is still empty stubs.

## Environment & commands

```powershell
.\env\Scripts\Activate.ps1
pip install -e .              # from the repo root; makes colo_project importable anywhere
```

Everything is a module under the `colo_project` package and runs from any working directory:

```powershell
python -m colo_project.scripts.run_mds --scenario test
python -m colo_project.scripts.run_nbp --scenario dense --radius 30
python -m colo_project.dataset.generate_dataset --scenarios-dir colo_project/dataset/scenarios --out-dir <dir>
```

Runs are **headless by default** (`matplotlib.use("Agg")`); pass `--show` to open figures. There are
no tests and no linter config; line lengths follow flake8 defaults (79).

**Every run is kept.** `io.result_dir` allocates a fresh `run_NNNN/` per invocation and never reuses
one, so a rerun of the same scenario/config/seed cannot overwrite an earlier result:

```
results/<scenario>_seed<n>/
    network.png  model_detection.png  model_rss.png   <- depend only on the scenario
    run_0001/    metrics*.json  arrays*.npz  figures/*.png
    run_0002/    ...
```

The index is `max(existing) + 1`, not a count, so deleting a run does not hand its number out again.
Figures go in `figures/`, never beside the json/npz.

## Data pipeline

`dataset/scenarios/<name>.json` → `generate_scenario()` →
`dataset/scenarios/<name>/<name>_seed<seed>.npz` → `Scenario.load()`.

`Scenario` (`dataset/data_loader.py`) is a frozen dataclass and **pure data** — it computes nothing
and plots nothing. Algorithms take a `Scenario` and return their own result object. Use
`load_or_generate(name, scenarios_dir)` rather than wiring paths by hand.

Deliberately *not* cached on `Scenario`: the n-hop distance graph. NBP wants it at a fixed
`cfg.n_hop` while MDS escalates the hop count until the graph is connected, so caching one would
force a shared knob onto two algorithms that disagree.

Scenario JSON keys: `num_nodes`, `num_anchors`, `d_dim`, `meters` (square field side, centred on the
origin), `radius` (comms range), `noise` (shadowing sigma in **dB**), `seed`, `placement` (true =
overwrite the first anchors with a hex grid, which also *changes* `num_anchors` to the grid count),
`heterogeneity` + `power_level` (per-node TX power band: `-1` uniform 0 dBm, `0` BLE, `1` Wi-Fi,
`2` cellular, `3` Zigbee, `4` RFID), `symetric`.

Two scenarios exist: `test` (4 random anchors, radius 20 — only 29 of 96 targets hear any anchor;
this is the *non-sufficient connectivity* regime and NBP does poorly on it by design) and `dense`
(7 grid anchors, radius 30 — well-posed; NBP reaches ~2.5 RMSE against an MDS baseline of ~3.0 and
a CRLB of ~0.93).

## Conventions that pervade the codebase

- **Anchors come first.** The first `num_anchors` rows of every `(N, ...)` array are anchors; the
  rest are targets. Prefer working with full-length `(N, ...)` arrays: anchor particles are copies
  of a known position, so a weighted belief mean over *all* nodes returns anchor positions exactly.
  That is why `nbp/core.py` never offsets an index by `num_anchors` — do not reintroduce
  `estimates[node - n_anchors]` style indexing.
- **Score on targets only.** Anchors are known, so including them dilutes RMSE with exact zeros.
- **Zero means "no edge", not "distance 0"** in `D`, `Dn` and `RSS`. Graph code must map `0 → inf`
  before shortest paths (`n_hop_distance` does) and must not treat masked entries as measurements.
- **`rng` is a required keyword-only argument** wherever randomness is drawn. There is no
  `rng=None` default anywhere, on purpose. Streams are derived as
  `np.random.default_rng([seed, STREAM_*])` (see `constants.py`), and NBP derives a further
  per-iteration stream, so changing `n_iter` cannot perturb the scenario and running MDS first
  cannot shift the draws NBP makes.
- **`scipy.stats.gaussian_kde.resample()` draws from the global legacy `np.random` singleton unless
  passed `seed=`.** Every call site must pass it or reproducibility silently breaks.
- **Path-loss constants live in `colo_project/constants.py`** (`ALPHA`, `D0`) and are written into
  each `.npz`. Read them off the `Scenario` rather than relying on function defaults, so the CRLB
  uses the parameters that actually generated the data.
- Noise is injected in the **RSS (dB) domain** as zero-mean Gaussian — which is standard log-normal
  shadowing, log-normal in linear power. Symmetrization mirrors the upper triangle rather than
  averaging with the transpose, because averaging would shrink the per-entry sigma by `1/sqrt(2)`.
  `metrics.range_sigma` converts a dB sigma into the ranging sigma at a distance, and is shared by
  the CRLB and the NBP proposal so the two cannot drift apart.
- **The CRLB is anchored.** Range-only measurements are invariant to translation and rotation, so
  the full `2N x 2N` FIM is rank-deficient by 3 and cannot be inverted. `metrics.crlb` drops the
  known anchor coordinates and inverts exactly; it returns a `(2*N_targets, 2*N_targets)` covariance
  and requires `num_anchors > 0`. Do not reintroduce Tikhonov regularization to invert the full FIM
  — that makes the "bound" a function of `eps` rather than of geometry, and MDS could beat it.
- Plot helpers **return a figure and never call `plt.show()`**; the caller saves or shows. They also
  all accept `ax=None` so they compose into a grid, and they take **absolute** node indices. Nodes of
  interest are picked from the data (worst error, widest prior) — the legacy prototype hardcoded node
  4 or 5 plus a matching zoom window, which is why none of its figures generalized.

## Module map (`colo_project/`)

- `constants.py` — `ALPHA`, `D0`, and the `STREAM_*` RNG stream ids.
- `dataset/` — `generate_dataset.py` (JSON → `.npz`), `data_loader.py` (`Scenario`, `load_or_generate`),
  `geo_positions.py` (real capture points → metres). `geo_frame` returns the projection reference,
  the recentring offset and the anchors-first `order` alongside the positions, so `unproject` can
  invert it; `load_geo_positions` is a thin wrapper over it, one code path so forward and inverse
  cannot drift apart.
- `utils/graph_utils.py` — the forward measurement model and n-hop graph construction.
  `n_hop_distance` is a dense min-plus DP, O(N^3) per hop, and dominates runtime for large N.
- `utils/metrics.py` — `euclidean_metrics`, `per_node_error`, `range_sigma`, anchored
  `crlb`/`jacobian`/`per_node_peb`, and `procrustes_disparity`/`procrustes_hist` (shape-only,
  scale-blind, dimensionless — see below).
- `utils/io.py` — `result_dir` (allocates `run_NNNN/`), `save_json`/`save_arrays`/`save_fig`,
  `git_sha` (resolved against this repo, not cwd).
- `utils/visualizations.py` — `plot_network`, `plot_results`, `plot_convergence`, `plot_error_cdf`,
  `plot_error_vs_degree`, `plot_particles`, `plot_messages`, `plot_detection_model`,
  `plot_rss_model`, `plot_raw_layouts` (unregistered layouts side by side) and
  `scenario_figures` (the scenario-only set, as `{stem: fig}`). There is deliberately no
  `plot_procrustes`: `plot_convergence` already takes a dict of curves plus flat baselines,
  which is the same figure the legacy prototype drew three separate times.
- `mds/classic_mds.py` — `run_mds` returns `(x_hat, affine, rigid)`; rigid (Procrustes) is
  substantially better than affine on these scenarios.
- `nbp/` — `bbox.py`, `potentials.py` (both pure, no RNG), `particles.py` (all RNG), `core.py`
  (`NBP`, `NBPConfig`, `NBPState`, `NBPResult`).
- `scripts/` — `run_mds.py`, `run_nbp.py`, `compare_procrustes.py` (raw MDS vs cold NBP vs
  warm-start NBP by Procrustes disparity; runs NBP twice, so ~2x the usual runtime),
  `export_geo.py` (a run's estimates → lat/lon, joined to the source records). `export_geo` needs anchors and refuses without them: an anchor-free estimate has
  no absolute frame to unproject through, and Procrustes-ing it onto the truth first would export
  the answer. `run_mds.py` likewise requires anchors — `metrics.crlb` and the MDS registration both
  do — so the `*_noanchor` scenarios are `run_nbp.py` only. `run_gnn.py` is an empty stub.

## NBP specifics

One iteration is two phases. **Phase A** builds, for every one-hop pair `r → u`, a weighted KDE over
the particles of r shifted onto the measured range annulus, weighted by detection probability and by
the *cavity* belief of r (`weights[r] / incoming[r, u]`, i.e. the belief of r with the previous
message from u divided out). **Phase B** multiplies each incoming message per node and resamples.

`incoming` is indexed `[receiver, sender]`: `incoming[u, r]` is the message `r → u` evaluated at the
particles of u.

`self.D` is the **n-hop** distance matrix and `self.C` is the **one-hop** adjacency. That split is
the core of the algorithm: one-hop neighbours send positive messages, while nodes reachable within
n hops that were *not* heard send negative information (`1 - E[detect]`). Do not collapse the two.

`NBPResult.extras` carries what the run would otherwise throw away, for the plots: `particles_hist`
`(n_iter, N, P, d)`, `weights_hist`, `spread_hist` `(n_iter, N_t)`, `bboxes`, and the final
iteration's `proposals` dict of KDEs (RAM only — a `gaussian_kde` does not go into an `.npz`, so the
message figure is its record). This is recording, not computing: nothing in it draws from `rng` and
nothing is inserted between two RNG consumers, so the trace stays bitwise identical. ~3 MB at N=100,
P=125, n_iter=10, linear in every knob — gate it behind a config flag before N reaches the thousands.

`cfg.warm_start` seeds the particles from a global layout (MDS unless the caller passes
`init_positions`) instead of from the anchor boxes or the whole field, with `cfg.warm_halfwidth`
as the seed box half-width (`None` -> `radius / 2`). It **replaces** the anchor bboxes rather than
intersecting with them, so warm-vs-cold stays a clean A/B; with anchors present the MDS layout was
registered using those anchors anyway. Off by default and a cold run is bitwise unchanged. This is
the `mds_init` legacy had, minus its two bugs: legacy made every particle an exact copy of the MDS
position (zero initial spread, degenerate first KDE) and overwrote the anchor rows while doing it.
Never seed from a truth-aligned layout -- `run_nbp` captures `mds_seed` *before* it replaces
`rigid` with the Procrustes-to-truth used for anchor-free scoring.

`metrics.procrustes_disparity` is the one score that compares *raw, unregistered* layouts: it
removes translation, rotation, reflection **and scale**, leaving shape agreement alone (`0` same
shape, `1` nothing in common — lower is better, despite the legacy name "similarity"). Because it
is scale-blind it cannot replace a metre-valued RMSE: ranges fix the scale here, so a uniformly
inflated layout is a real error that disparity scores as perfect. Report the two together.
`compare_procrustes.py` does, and the gap between them is informative — on `paris_noanchor` warm
NBP beats the MDS layout it was seeded from on disparity (0.0013 vs 0.0073) while *losing* to it
on aligned RMSE (10.56 m vs 10.16 m), because its shape is better but it sits 9% too large.
Cold and anchor-free, disparity stays at 0.89-0.99: since scale is already divided out, that says
the cold failure is not merely the extent collapse but a scrambled geometry.

Anchor-free (`num_anchors == 0`) runs go through `metrics.crlb_anchor_free` and `align_rigid`
rather than `crlb`: with no anchors the FIM is singular by exactly 3 (two translations plus
rotation, written down analytically in `gauge_basis`), so the bound is taken on the orthogonal
complement of that nullspace, and error is scored after a free Procrustes as well as raw. Cold and
anchor-free, NBP collapses to ~1/6 of the true extent on iteration 1 and does not recover; see
`Report2.md` for why, and do not reach for a larger `n_hop` as the fix.

Known behaviour, not a bug: RMSE bottoms out around iteration 5 and drifts up slightly afterwards
(particle depletion / overconfidence). Runtime is ~25-60s for N=100, P=125, 10 iterations; the hot
path is `gaussian_kde`, and the negative-information block is the thing to vectorize first if that
ever matters. Any optimization must preserve the bitwise-identical trace at a fixed seed.

## Legacy `NBP/` (reference only, never modified)

`optimized_NBP.py` is the version the port came from; `NBP_iteration2` there is dead but is the
better-documented reference. `_COLO.py` (2.9k lines) is its helper grab-bag. Capabilities
deliberately **not** ported: the multi-config sweep runner (`run_experiment`),
`error_vs_neighborhood`, and the weighted SMACOF / spectral-layout MDS baselines in `Test.py`.
(`mds_init` *was* ported, as `NBPConfig.warm_start`; Procrustes similarity tracking *was* ported,
as `metrics.procrustes_disparity` plus `scripts/compare_procrustes.py` — both above.) The
legacy results dict reported `var(D_noisy - D_clean)` under the key `"CRLB"` — a ranging variance
in m^2, plotted against a positioning RMSE in m. It is not a bound; use `metrics.crlb`.
