# TODO

## Real-location dataset -> simulated RSS -> NBP

Berk has a downloaded dataset of images with their locations. The plan:

1. Take the locations from the dataset as node positions.
2. Place an RSS source at each location and simulate RSS values.
3. Invert RSS to distances.
4. Run NBP on the resulting graph.

Steps 2-4 are already what `colo_project` does. The only genuinely new part is
step 1: swapping the *source* of `X_true` from synthetic to real.

### The one seam that needs adding

`dataset/generate_dataset.py:generate_scenario()` currently gets positions from

```python
X_true, area = generate_targets(num_nodes, d_dim, meters, rng=rng)
```

Everything after that line — `get_distance_matrix` (path loss, shadowing,
radius thresholding), the `.npz` contract, `Scenario`, MDS, CRLB, NBP — is
position-source agnostic and should work unchanged. So this wants an
alternative branch that loads `X_true` from the dataset instead, e.g. a
`positions` key in the scenario JSON pointing at a file, with `generate_targets`
as the fallback when it is absent.

### Gotchas already known from the current code

- **Coordinates must be centred on the origin.** `generate_targets` returns a
  field centred at 0, and `NBP` builds its bbox limits as
  `[-meters/2, +meters/2]` per axis from `NBPConfig.meters`. Real coordinates
  will not be centred and probably will not be square. Either translate them to
  the origin at load time, or carry real bounds through instead of `meters`.
- **Units.** `radius`, `d0` and the resulting distances are metres. Lat/lon has
  to be projected to a metric frame first, and the extent decides whether
  `radius` and `ALPHA`/`D0` are still sensible.
- **Anchors come first.** The first `num_anchors` rows of `X_true` are the
  anchors, so whichever real locations act as anchors have to be sorted to the
  front. `placement: true` overwrites them with a hex grid and will fight a real
  layout — set it `false`.
- **`radius` drives everything.** It thresholds connectivity *and* is the scale
  of the detection kernel in NBP. Real spacing will not match the current 20/30
  m; the ablation showed connectivity dominates every algorithmic effect
  measured so far.
- Scenario `seed` stays meaningful — positions become fixed, but the shadowing
  draw is still random, so repeated seeds still give repeated noise.

---

## Computational complexity — investigate later

Not urgent at N=100. Becomes the blocker the moment the real dataset is larger,
so measure before scaling up rather than after.

### `utils/graph_utils.n_hop_distance` — the hard wall

Dense min-plus DP. Per hop it materialises `D_prev[:, :, None] + W[None, :, :]`,
an **N^3 temporary**. Measured, 2 hops:

| N | time | peak temporary |
|---|---|---|
| 100 | 0.003 s | 0.01 GB |
| 200 | 0.022 s | 0.06 GB |
| 300 | 0.079 s | 0.22 GB |
| 400 | 0.180 s | 0.51 GB |

Clean N^3 scaling. **Memory dies long before time does**: 8 GB at N=1000, 64 GB
at N=2000. Time at N=400 is still only 0.18 s, so this is an allocation problem,
not an arithmetic one.

Fix direction: the graph is sparse (mean degree ~10-21), so
`scipy.sparse.csgraph.dijkstra` / `shortest_path` gives the same answer without
the N^3 temporary. The one wrinkle is that the current function bounds paths by
*hop count*, which plain Dijkstra does not — needs either a hop-limited variant
or, since the ablation showed nothing above 2 hops is worth having, just
cap at 2 and use a sparse product.

Callers: `NBP.__init__` (once per run, at `cfg.n_hop`) and
`ClassicMDS.run_mds`, which escalates the hop count in a loop until the graph is
connected — so MDS pays this repeatedly and is the worse offender.

### NBP per-iteration cost

Measured ~25-60 s for N=100, P=125, 10 iterations. The hot path is
`scipy.stats.gaussian_kde` (construct, resample, evaluate), whose Cython kernel
does not release the GIL — which is why the port is serial.

Scaling is roughly O(pairs x M x P) with `pairs` growing ~N^2, so N=1000 is
~100x the message work: hours, not minutes. Two things to look at, in order:

1. The negative-information block is an (M, P) pairwise distance per push pair,
   and hop 4 on dense already has 7702 push pairs. Vectorise across all
   non-neighbour senders into one `cdist` rather than looping.
2. `relative_spread` fits a KDE on `3P` extended angle samples **per edge per
   iteration**, which is easy to miss and probably the second cost.

Whatever changes, the trace must stay bitwise identical at a fixed seed — that
is what the determinism work is for.

### Open questions, for whenever this starts

- Are the locations metric (x, y) or geographic (lat, lon)?
- What is the spatial extent, and how many nodes? `n_hop_distance` is O(N^3) per
  hop, so N in the thousands needs a sparse shortest-path instead.
- Which nodes are anchors — a chosen subset of real locations, or synthetic ones
  added on top?
- What role do the *images* play? Nothing in the NBP path consumes them; they
  may be for the (currently empty) `gnn/` stage rather than this one.
- Is there any real RSS in the dataset, or is it fully simulated from the
  path-loss model?
