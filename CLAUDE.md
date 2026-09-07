# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **cooperative localization (COLO)** in wireless sensor networks: estimating node
positions from noisy RSS-derived pairwise range measurements, using **Nonparametric Belief
Propagation (NBP)** with bounded-box priors, plus a classic-MDS baseline and a planned GNN approach.
Reference papers are in `Sources/`.

The repo is mid-refactor. `NBP/` is the legacy working prototype; `colo_project/` is the clean
rewrite it is being ported into. Most of `colo_project/colo/`, `colo_project/gnn/`, and
`colo_project/scripts/` are still empty stubs — the NBP port is the open work item (see the
`MDS done. TODO: NBP` commit).

## Environment & commands

Virtualenv lives at `env/` (Python 3.14, gitignored):

```powershell
.\env\Scripts\Activate.ps1
pip install -r colo_project\requirements.txt
```

**All commands must run with `colo_project/` as the working directory, and as modules (`-m`).**
Imports are top-level absolute (`from utils.graph_utils import ...`) and `pyproject.toml` is empty,
so nothing is installed as a package — `python dataset/generate_dataset.py` fails with
`ModuleNotFoundError: No module named 'utils'`.

```powershell
cd colo_project

# Generate .npz scenarios for every JSON config in a directory
python -m dataset.generate_dataset --scenarios-dir dataset/scenarios --out-dir dataset/scenarios/out

# Generate + load + run the MDS pipeline for scenarios/test.json (opens matplotlib windows)
python -m dataset.data_loader
```

There are no tests, no linter config, and no build step. Line lengths follow flake8 defaults (79).

## Data pipeline

`dataset/scenarios/<name>.json` → `generate_scenario()` → `dataset/scenarios/<name>/<name>_seed<seed>.npz`
→ `LocalizationScneario(npz_path)`.

The `.npz` holds the full ground truth of one experiment and is the contract between generation and
every algorithm: `X_true` (N×d), `full_D` (true pairwise distances), `D` (noisy distances, zeroed
beyond `radius`), `B` (binary adjacency), `RSS` (masked by `B`), `num_anchors`, `noise`.

Scenario JSON keys (see `generate_dataset.py` for defaults): `num_nodes`, `num_anchors`, `d_dim`,
`meters` (square field side, centred on the origin), `radius` (comms range), `noise` (RSS sigma in
dB), `seed`, `placement` (true = overwrite the first anchors with a hex grid), `heterogeneity` +
`power_level` (per-node TX power drawn from a band: `-1` uniform 0 dBm, `0` BLE, `1` Wi-Fi,
`2` cellular, `3` Zigbee, `4` RFID), `symetric` (symmetrise RSS and its noise).

## Conventions that pervade the codebase

- **Anchors come first.** The first `num_anchors` rows of `X_true` (and of every particle/estimate
  array) are known-position anchors; everything after is a target to localize. Slices like
  `X_true[num_anchors:]` and `estimates[node - n_anchors]` are everywhere — off-by-one here is the
  most common bug class in this code.
- **Zero means "no edge", not "distance 0".** `D` and `RSS` are masked, so any graph algorithm must
  convert `0 → inf` before shortest paths (`n_hop_distance` does this) and must not treat masked
  entries as measurements.
- **Path-loss constants `alpha=3.15`, `d0=1.15` are duplicated as defaults** in
  `utils/graph_utils.get_distance_matrix`, `utils/metrics.crlb`, and `utils/metrics.jacobian`.
  Changing the forward model means changing all three, or the CRLB stops matching the data.
- Noise is injected in the **RSS domain** (`RSS_to_distance` adds lognormal noise in dB), then
  inverted back to distance — so distance error grows with range. The CRLB in `metrics.crlb`
  mirrors this: per-link distance sigma is `(ln10 / (10·alpha)) · d_ij · sigma_db`.
- `LocalizationScneario` (typo is the real class name) is the single entry point that loads an
  `.npz` and eagerly computes n-hop graphs, the CRLB covariance, per-node PEB, and — when
  `run_mds=True` — the whole MDS solution. Constructing it has side effects (prints, matplotlib
  windows).

## Module map (`colo_project/`)

- `dataset/` — `generate_dataset.py` (JSON → `.npz`), `data_loader.py` (`.npz` → `LocalizationScneario`).
- `utils/graph_utils.py` — the forward measurement model (`get_distance_matrix`,
  `distance_to_RSS`, `RSS_to_distance`), node/anchor placement, and n-hop graph construction
  (`n_hop_distance` is a dense min-plus DP, O(N³) per hop — it dominates runtime for large N).
- `utils/metrics.py` — `euclidean_metrics` (RMSE/MAE/MedAE), `crlb`/`jacobian`/`per_node_peb`
  (the theoretical error floor plotted against algorithm RMSE).
- `mds/classic_mds.py` — `ClassicMDS.run_mds` is the baseline: raise the hop count until the n-hop
  distance graph is fully dense, classic MDS embed, then anchor registration (affine via pseudo-
  inverse, or rigid via orthogonal Procrustes). MDS output also seeds NBP initialisation.
- `colo/` — the NBP port target: `bbox.py`, `particles.py`, `potentials.py`, `nbp.py` (all empty).
- `utils/visualizations.py` — `plot_MRF` (network graph), `plot_results` (true vs. estimated, with
  error lines). Both call `plt.show()` and block.

## Porting from `NBP/` (legacy)

`NBP/optimized_NBP.py` is the reference implementation to port; `NBP/_COLO.py` (2.9k lines) is the
grab-bag of helpers it imports via `from _COLO import *`. The pieces that map onto the empty
`colo/` stubs:

- `bbox.py` ← `create_bbox`: intersect per-anchor axis-aligned boxes of half-width `D[i,j]` to bound
  each target's feasible region.
- `particles.py` ← `generate_particles`: uniform samples inside each box; anchors get `n_particles`
  copies of their own position.
- `potentials.py` ← `_COLO.mono_potential_bbox` (uniform prior over a box) and `_COLO.duo_potential`
  (Gaussian pairwise range likelihood).
- `nbp.py` ← `NBP.iterative_NBP` / `NBP.NBP_iteration`: per iteration, approximate each message
  `r→u` by shifting node r's particles, weighting by detection probability and the reciprocal of
  the reverse message, fitting a `gaussian_kde` proposal and resampling; then multiply incoming
  messages per node to update the belief and resample to `n_samples`. Non-neighbours within n-hop
  range contribute *negative* information (`1 - Σ w·p_detect`). Estimates are the weighted particle
  means (`np.einsum('ijk,ij->ik', ...)`).

Note the legacy code carries a different, incompatible `get_distance_matrix` signature and its own
`generate_targets`/`RMSE` — port the algorithm, not the helpers; `colo_project/utils/` already has
the current versions.

`colo_project/save.txt` is a scratch snippet of a previous MDS wiring, not live code.
