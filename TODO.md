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
