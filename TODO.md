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

---

## `relative_spread` bearing bandwidth — fix measured, then reverted

`nbp/particles.py:relative_spread` replicates the bearing samples at `+/-2pi`
before fitting `gaussian_kde`, so the wrap-around at `+/-pi` is not treated as a
hard boundary. That works, but scipy sizes its bandwidth as *factor x the
standard deviation of whatever it is handed*, and the replicas inflate that std
from the bearing spread (~0.13 rad) to ~5.1 rad. The kernel is therefore ~29x
wider than the data it is smoothing.

Instrumented over 3892 real calls on `dense`, iterations 2-3:

| quantity | median |
|---|---|
| belief bearing spread (circular std) | 0.131 rad (7.5 deg) |
| Silverman bandwidth on the raw angles | 0.057 rad (3.3 deg) |
| bandwidth as coded (replicated) | 1.664 rad (95.3 deg) |
| inflation | **29.3x** |

Bandwidth exceeds the bearing spread on 99.6% of calls, so the "bearing-aware"
proposal is close to uniform over the circle — in a synthetic check 54% of
proposed angles landed more than 45 deg off the true bearing. This costs
sampling efficiency only, not correctness: `w_xy` is returned and divided out at
`core.py:199`, so the target is still right.

### What the fix did

Sizing the kernel from the unreplicated angles and passing it back as a scalar
factor reproduces scipy's Silverman bandwidth exactly, and the proposed bearing
spread drops from 74 deg to 8.8 deg against a true 8.6 deg. Per-iteration RMSE,
3 seeds:

```
dense_s0  before   8.34  4.02  3.23  2.84  2.81  2.93  3.07  3.03  3.18  3.15
          after    8.34  4.25  3.18  2.77  2.69  2.62  2.55  2.53  2.50  2.49
dense_s1  before   8.69  5.35  4.68  4.00  3.61  3.36  3.23  3.21  3.20  3.32
          after    8.69  6.85  6.51  6.33  6.33  6.24  6.17  6.09  6.05  6.02
dense_s2  before   8.85  5.55  4.70  3.98  3.45  3.34  3.27  3.31  3.32  3.32
          after    8.85  6.69  6.08  5.92  5.86  5.78  5.72  5.64  5.60  5.56
```

Mean best RMSE on `dense` went 3.09 -> 4.69. `test` was a wash (best 34.38 ->
34.45, final 35.26 -> 34.95), but that scenario fails outright at ~34 m on a
100 m field, so it carries no signal either way.

### Why it got worse — the finding worth keeping

The oversmoothing is acting as **unintentional exploration**. Two effects, both
visible above:

- Mean belief spread on `dense` collapses from a flat ~1.55 to 0.20 and is still
  shrinking at iteration 10. The clouds stop wasting particles on bearings the
  belief rejects, and concentrate hard.
- The drift documented in `CLAUDE.md` ("RMSE bottoms out around iteration 5 and
  drifts up slightly afterwards") **disappears**: every curve is monotone
  decreasing, and seed 0 reaches 2.487, better than its old best of 2.808 and
  still descending.
- But seeds 1 and 2 jump to a worse configuration at iteration 2 and never
  recover, because spread is already 0.49 by iteration 3. The beliefs end
  badly inconsistent: spread 0.20 against an actual error of 6.02.

So the near-uniform proposal was rescuing bad early iterations by continually
injecting particles in rejected directions — and was simultaneously what capped
seed 0 at 2.81 and caused the late drift. Fixing the bandwidth buys efficiency
and premature convergence in the same stroke.

### To try, in order

1. Keep the corrected bandwidth but reintroduce the smoothing **explicitly** as
   an oversmoothing factor on the angle spread, then sweep it (1, 2, 4, 8, 16x
   over 3 `dense` seeds). The optimum is somewhere between 1x (premature) and
   29x (over-diffuse); seed 0's 2.487 says there is real headroom below the
   current 3.09 baseline. No one can tune around a bug, which is the main
   argument for doing this at all.
2. Or leave the proposal at 1x and cure impoverishment at its source, with
   roughening / jitter at the resample step in `_update_beliefs`. That is the
   principled fix; the proposal width is a hack that happens to work.
3. Unexplained and worth a second look: runtime on `dense` rose from a
   consistent 64-67 s to 87-151 s after the fix (baseline re-timed afterwards to
   rule out thermal effects). Bandwidth should not affect `gaussian_kde` cost.

Related, and *not* a bug: pairing u's particle `i` with r's particle `i` in
`relative_spread` looks like it asserts a correspondence between clouds, but the
paired differences only ever reach a 1-D KDE, which uses the marginal. The two
clouds are resampled through independent generators, so index `i` carries no
cross-node meaning and any coupling gives the same fitted density — verified
against a permuted coupling and against all `P^2` pairs. Using all pairs would
only reduce the variance of the fit.
