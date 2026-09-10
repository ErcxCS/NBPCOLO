# Shape without a frame: Procrustes similarity, and what `n_hop` does to the distance matrix

Follow-up to `Report2.md`. Same two anchor-free real datasets (`barcelona_noanchor`,
`paris_noanchor`), same seed (31), same radii. Two questions, one measurement each:

1. How close is the n-hop distance matrix to the true one, at 1 hop, 2 hops, and at
   the hop count where it finally becomes *complete*?
2. With no anchors and nothing aligned, how close in **shape** are the layouts that
   raw MDS, cold NBP and warm-start NBP produce?

**Headline: the n-hop matrix is systematically *short*, not long, and buying full
connectivity costs 3.6-4.6x its accuracy. `n_hop` turns out to be the knob that controls
the cold anchor-free collapse — at 1 hop NBP shrinks to a fifth of the true extent,
at 8 hops it recovers 87% of it — but even fully connected, cold NBP's shape is still
80-400x worse than a warm start's, so `n_hop` is a mitigation and initialisation is
still the fix.**

---

## 1. What was measured, and why it needed a new metric

With no anchors, MDS, cold NBP and warm NBP each land in their own arbitrary frame.
A metre-valued RMSE across the three measures three different gauges, not three
estimates. `metrics.procrustes_disparity` divides out translation, rotation,
reflection **and scale** and returns what survives, which is shape alone: `0` for
identical shapes, `1` for nothing in common.

Everything below is scored on layouts exactly as the estimator produced them — no
Procrustes onto the truth, no anchor registration, nothing moved. The disparity
handles the gauge internally, which is what makes the comparison possible at all.

The one thing disparity cannot see is scale, and here that matters: ranges fix the
absolute size of the network, so a uniformly inflated layout is a real error that
scores a perfect 0. Every disparity below is therefore reported next to the layout's
extent (rms radius / truth's rms radius). The two together say more than either.

`utils/graph_utils.hops_to_complete` finds the hop count at which every pair becomes
reachable — the `n` classic MDS escalates to, since it cannot work around holes.
**Both datasets complete at `n_hop = 8`**, despite mean degrees above 20.

---

## 2. The distance matrix against ground truth

`utils/graph_utils.compare_hop_distances`, against `full_D` (true clean distances),
over the pairs covered at each hop:

**barcelona_noanchor** (N=153, mean degree 20.9)

| hop | coverage | bias | RMSE | rel. bias | new pairs (their bias) | direct edges overwritten |
|----:|---------:|-----:|-----:|----------:|-----------------------:|-------------------------:|
| 1 | 13.7% | −0.30 m | 2.05 m | −0.5% | — | 0.0% |
| 2 | 35.2% | −1.02 m | 3.04 m | −2.1% | 5 002 (−1.13 m) | 28.4% |
| 8 | 100.0% | −3.46 m | 7.46 m | −3.0% | 15 060 (−4.36 m) | 29.2% |

**paris_noanchor** (N=200, mean degree 21.7)

| hop | coverage | bias | RMSE | rel. bias | new pairs (their bias) | direct edges overwritten |
|----:|---------:|-----:|-----:|----------:|-----------------------:|-------------------------:|
| 1 | 10.9% | −0.43 m | 2.58 m | −0.8% | — | 0.0% |
| 2 | 29.3% | −1.11 m | 4.03 m | −2.0% | 7 310 (−1.01 m) | 36.0% |
| 8 | 100.0% | −0.03 m | 11.82 m | −0.8% | 28 148 (+0.89 m) | 37.1% |

### 2.1 The bias is negative, which is the opposite of the geometric expectation

A path through intermediate nodes is a polyline, and a polyline is never shorter than
the straight line it spans. On noiseless data multi-hop distances can therefore only
be **over**-estimates. Measured, they are under-estimates at almost every hop count.

The reason is that the min-plus DP takes a *minimum over many noisy path sums*, and a
minimum over noisy candidates preferentially selects the candidates whose noise ran
negative. That selection effect is an extreme-value bias, it grows with the number of
alternative paths, and on these datasets it beats the geometric one.

The cleanest evidence is the last column: **28-37% of measured one-hop edges are
overwritten by a detour the DP thinks is shorter.** Those are pairs with a direct
measurement, where geometry guarantees no detour is genuinely shorter — so every one
of those replacements is noise winning. This is exactly the failure `NBP` already
guards against by keeping the direct range for one-hop pairs in `D_hop` (commit
`f505e36`, which measured ~31% on `dense`); the numbers here confirm it at N=153-200
on real geometry, and confirm that the fraction is a property of the noise and degree
rather than of `dense` in particular.

Paris at hop 8 is the interesting exception: bias ≈ 0 (−0.03 m) while RMSE is the
worst in the table (11.82 m). The two biases have cancelled *in the mean* while the
spread kept growing. A near-zero mean error is not evidence of a good matrix here,
which is why the table reports both.

### 2.2 Completeness is expensive

Going from 2 hops to full connectivity multiplies error RMSE by **2.5x** (barcelona)
and **2.9x** (paris); against 1 hop it is **3.6x** and **4.6x**. Coverage rises from
~30% to 100%, but the entries that fill the gaps are the worst ones — barcelona's
15 060 newly-covered pairs carry a −4.36 m bias against the −1.02 m of the pairs
already there. Runtime rises steeply too: the hop-8 runs took several times longer
than hop 2, since `n_hop_distance` is an O(N³) pass per hop *and* ~100% pair coverage
puts every pair into NBP's negative-information block.

---

## 3. Procrustes similarity of the layouts

Disparity `M²` against ground truth (lower is better), raw and unaligned, alongside
each layout's extent as a fraction of the truth's:

**barcelona_noanchor**

| n_hop | MDS raw | NBP cold | NBP warm | cold extent | warm extent |
|------:|--------:|---------:|---------:|------------:|------------:|
| 1 | 0.01164 | 0.95933 | 0.00237 | 0.236 | 0.930 |
| 2 | 0.01164 | 0.89140 | **0.00216** | 0.466 | 1.047 |
| 8 | 0.01164 | **0.24527** | 0.00299 | 0.874 | 1.093 |

**paris_noanchor**

| n_hop | MDS raw | NBP cold | NBP warm | cold extent | warm extent |
|------:|--------:|---------:|---------:|------------:|------------:|
| 1 | 0.00725 | 0.98500 | 0.00149 | 0.213 | 0.994 |
| 2 | 0.00725 | 0.98692 | **0.00127** | 0.399 | 1.092 |
| 8 | 0.00725 | **0.68796** | 0.00170 | 0.695 | 1.126 |

**MDS is identical across all three rows**, as it must be: `ClassicMDS` escalates its
own hop count until the graph is connected and never reads `cfg.n_hop`. That column is
the control, and it holding constant is what says the sweep varied only what it meant to.

### 3.1 `n_hop` controls the cold collapse

`Report2.md` recorded that cold anchor-free NBP collapses to a fraction of the true
extent and does not recover. The extent columns show that fraction is a *function of
`n_hop`*: 0.236 → 0.466 → 0.874 on barcelona, 0.213 → 0.399 → 0.695 on paris. At one
hop the estimate is a fifth of the network; fully connected, it is most of it.

Disparity improves alongside it — 0.959 → 0.891 → 0.245 — and because disparity is
scale-blind, that improvement is **not** the extent recovery being counted twice. The
shape genuinely gets better. With only 1-2 hops of reach, the pull term has no
information about far-apart node pairs at all, so nothing resists the contraction; a
complete matrix constrains every pair and the layout has to spread out to satisfy it.

So `CLAUDE.md`'s standing advice — *"do not reach for a larger `n_hop` as the fix"* —
needs one qualification. It is right that `n_hop` is not the fix: at hop 8 cold NBP
still sits at 0.245, which is **~80x worse than the warm start's 0.00299 and ~20x
worse than plain MDS's 0.01164**, for several times the runtime and a distance matrix
with 2.5x the error. But it is not inert either, and describing the collapse as
insensitive to `n_hop` would be wrong.

### 3.2 Warm start dominates everywhere, and is nearly flat in `n_hop`

Warm-start disparity moves only between 0.00127 and 0.00299 across the whole sweep,
while cold moves by a factor of four. Seeded with a global layout, NBP does not need
the hop graph to tell it about far-apart pairs — it already has a guess — so the thing
`n_hop` was buying is redundant. Warm start beats the MDS layout it was seeded from by
**3.9-5.7x on every row**, so it is genuinely refining the seed, not merely
preserving it.

The best warm result is at **hop 2** on both datasets, not at full connectivity. Hop 8
is worse (0.00299 / 0.00170), which is consistent with §2: the complete matrix is the
least accurate one, and an estimator that no longer needs its extra coverage only
inherits its extra noise.

### 3.3 Scale is where warm start is weakest, and hop count trades against it

Warm extent runs 0.930 → 1.047 → 1.093 (barcelona) and 0.994 → 1.092 → 1.126 (paris):
more hops, more inflation, up to 13% too large. Since ranges fix the scale, that is a
real error disparity cannot see — and it is what makes warm NBP lose on metre-valued
aligned RMSE to MDS on paris (10.56 m vs 10.16 m) while winning 5.7x on disparity.

Note the two metrics prefer different hop counts: shape is best at hop 2, scale is
best at hop 1 (0.930 / 0.994, the closest to unity in the table). Reporting only one
of them would hide the trade.

---

## 4. What to take from this

- **The n-hop matrix is biased short, not long.** Anything that reasons about
  multi-hop ranges as conservative over-estimates is reasoning from the noiseless
  geometry and will be wrong by a few percent in the other direction.
- **~30-37% of direct measurements get overwritten by noise-shortened detours.**
  `NBP` already protects against this; MDS does not, and consumes the raw n-hop matrix
  at whatever hop count connectivity demands.
- **Completeness is not free.** 8 hops is what these datasets need, and it costs 2.5-2.9x
  the distance error versus 2 hops.
- **`n_hop` mitigates the cold anchor-free collapse but does not fix it.** Warm start
  is ~80x better at a fraction of the cost.
- **Report disparity and a metre score together.** Every place they disagreed here
  (paris warm vs MDS; hop 8 vs hop 2), the disagreement was scale, and it was real.

## 5. Reproducing

```powershell
python -m colo_project.scripts.compare_procrustes --scenario barcelona_noanchor `
    --radius 40 --meters 264.3933 --n-hop 2
python -m colo_project.scripts.compare_procrustes --scenario paris_noanchor `
    --radius 50 --meters 322.7597 --n-hop 2
```

`--n-hop 1` and `--n-hop 8` give the other rows. `run_nbp` does not derive `radius`
and `meters` from the scenario, so the geo datasets need them passed explicitly;
`dataset.geo_positions.geo_frame(...)["meters"]` is where the values come from.

The distance-matrix tables are `graph_utils.compare_hop_distances(sc.D, sc.full_D,
[1, 2, hops_to_complete(sc.D)])`.

Each run writes `metrics_procrustes.json` (disparity, per-iteration history, raw
RMSE), `arrays_procrustes.npz` (the three raw layouts) and two figures —
`procrustes_layouts.png`, the three unaligned layouts side by side, and
`procrustes_tracking.png`, disparity per iteration on a log axis.

**Caveat: one seed.** Every number here is seed 31 on two datasets. The large effects
(the negative bias, the direct-edge overwrite rate, cold-vs-warm) are far too big to
be seed noise, but the small ones — hop 2 versus hop 8 for warm start, differing in
the fourth decimal — are not established by this and should not be quoted as an
ordering without a seed sweep.
