# Research ideas

Algorithm changes, not refactoring. Nothing here is implemented. Each entry
records the evidence for and against so it does not have to be re-derived.

Measurements below are on `dense` (100 nodes, 7 grid anchors, R=30, sigma 1 dB)
and `test` (100 nodes, 4 random anchors, R=20), both seed 31, at commit 51411f1.

---

## 1. Push AND pull on the same multi-hop pair

**Status:** plausible, untested. Proposed by Berk, 2026-09-07.

Send a multi-hop pair *both* messages instead of one or the other: push it
beyond `R` (it was not heard) and pull it toward `D_hop[u,r]` (the path says
roughly how far), with `R < d <= k*R` as a sanity envelope.

### Why the earlier nth_n=2 experiment is not this experiment

Berk previously set `nth_n=2, n=4`, making 2-hop pairs "immediate" so they got
pull messages at the 2-hop pseudo-range. It did not work.

The cause is structural, not a property of the idea. In the legacy code `C` is
an **exclusive switch**: Phase A requires `C[r,u] == 1` to send a pull, and
Phase B sends a push only in the `else`. A pair is one or the other, never both.
So `nth_n=2` did not *add* the pull, it **traded away the push** on exactly the
2-hop pairs.

The hop ablation (commit 51411f1) says that trade is catastrophic: on dense the
push is worth 10.08 -> 2.98 RMSE, essentially the whole algorithm, and it is the
hop-2 pairs that deliver it. Giving up ~7.6 m of push to gain ~2 m of range
information is a bad trade, and the failure is what one should predict.

Applying both terms to the same pair is a configuration the legacy code cannot
express. It has not been tested.

### The pseudo-ranges are good, and best where the graph is dense

For pairs at exactly 2 hops, `D_hop[u,r]` against the true distance:

| | dense (R=30, deg 20.8) | test (R=20, deg 9.6) |
|---|---|---|
| mean true distance | 43.7 m | 28.4 m |
| `D_hop / d_true` | 0.983 | 1.031 |
| absolute error | **1.93 m** | 1.97 m |
| error a *direct* link would have at that range | 2.55 m | 1.66 m |
| ratio | **1.32x better** | 0.84x (worse) |

On dense a 2-hop pseudo-range is *more* accurate than a direct measurement would
be, and at 3 hops 1.50x more. Two causes: `range_sigma` scales linearly with
distance, so two 22 m legs beat one 44 m leg, and min-over-many-paths cuts
variance further. On test, with roughly half the degree, there are fewer paths
to min over and it comes out ~20% worse than a direct link. So this idea should
be expected to help on well-connected graphs and do little on sparse ones.

### The envelope is nearly a no-op

Measured on exactly-2-hop pairs: `D_hop > 2R` occurs **0.0%** of the time, and
`D_hop < R` 4.6% (dense) / 2.1% (test). The bounds almost never bind. Keep them
as a cheap clamp on the ~4% of provably corrupt pseudo-ranges, but the substance
of this idea is the annulus at `D_hop`, not the inequality.

### The gap it addresses

For a pair with `C[u,r] == 0` but reachable within n hops, the current message is

```
1 - E_r[ exp(-||x_u - p_r||^2 / 2R^2) ]
```

which increases monotonically with distance and saturates at 1. It says only
*"you are probably not within R of me"* and never *"but you are not arbitrarily
far either."* The multi-hop pseudo-range in `D_hop[u,r]` is used purely as a
gate for whether the pair participates; its value is never read. Legacy did the
same, so this is not something the port dropped.

The real deficiency is not the missing ceiling, it is that **a one-sided
constraint has no triangulating power.** Every "be far from r" message is
simultaneously satisfied by fleeing in any direction, so N of them do not pin a
position. An annulus does: a node holding 30 loose annuli from 30 different
neighbours is genuinely multilaterated even though each annulus alone is weak.
That is the case for this change.

### Use the hop count, not the pseudo-range

Two candidate upper bounds were measured. They behave completely differently.

**`d <= D_hop[u,r]` (triangle inequality) — does not work.** A path sum is
mathematically >= the straight-line distance, but `n_hop_distance` takes a *min
over paths*, so on a dense graph it selects the luckiest noise draw and lands
*below* the true distance:

| fraction of multi-hop pairs where `D_hop >= d_true` | hop 2 | hop 3 |
|---|---|---|
| dense | 33.4% | 19.6% |
| test | 54.8% | 57.6% |

Mean slack is *negative* on dense (-0.75 m at hop 2, -1.78 m at hop 3). The
denser the graph the more paths there are to take a min over, so the bound gets
worse exactly where the graph is best. Do not use it.

**`d <= k*R` for a pair at exactly k hops — works.** Purely geometric, no
measurement involved, so noise cannot corrupt it:

| exactly k-hop pairs | k=2 | k=3 | k=4 |
|---|---|---|---|
| dense: `P(d <= kR)` | 98.8% | 99.6% | 100% |
| test: `P(d <= kR)` | 99.5% | 99.8% | 100% |
| dense: `P(d > R)` | 96.8% | 100% | 100% |
| test: `P(d > R)` | 97.0% | 100% | 100% |

Both sides hold ~99% of the time across scenarios and hop counts. Note this
needs the *exact* hop count per pair, which `n_hop_distance` does not currently
return — it would come from the first `h` at which reachability turns on.

### Shape

```
msg(x_u) = E_r[ (1 - detect(d, R)) * annulus(d; D_hop[u,r], c*sigma_path) * clamp(d, R, k*R) ]
```

with `d = ||x_u - p_r||`. The first factor is today's push, unchanged. The
second is the new pull. The third is the clamp, soft rather than hard since ~1%
of pairs genuinely violate it.

`sigma_path` is the accumulated ranging sigma along the path, roughly
`sqrt(sum_i range_sigma(leg_i)^2)`; `c >= 1` is a trust factor that down-weights
the pull to account for double counting (see below). `c -> inf` recovers current
behaviour exactly, which makes this a sweepable knob rather than a yes/no.

Cost is near zero: `negative_information` already forms the full pairwise
`diff_sq`, so both new factors are elementwise terms on a matrix already built.
It does need the *exact* hop count per pair and the path leg lengths, neither of
which `n_hop_distance` currently returns.

### Objections

- **Double counting.** `D_hop[u,r]` is a deterministic function of the same
  one-hop measurements already entering as direct messages, so it is not
  independent evidence. On dense this would add 1805 positive messages on top of
  2084 real ones. Loopy BP will get more overconfident: expect `spread` to
  tighten faster than RMSE improves, and watch both. The `c` trust factor is the
  mitigation, and is the main thing to sweep. This is probably why the original
  authors used only the binary non-detection, which *is* a genuinely new
  observation rather than a derived one.
- **It will likely not rescue `test`.** The hypothesis that the push makes
  estimates flee outward is **false**: they are *under*-dispersed, collapsed
  toward the field centre by the bbox prior (mean `|est|` 11.6 / 14.6 / 18.1 at
  hop 1/2/3 against a true 25.4). The push expands them *toward* the correct
  spread and RMSE still worsens, so the `test` failure is wrong azimuth, not
  wrong magnitude. A ceiling would mostly be inactive there. Any benefit has to
  come from the triangulation argument, not from bounding a runaway.
- Related asymmetry worth fixing regardless: the one-hop path divides out the
  cavity (`weights[r] / incoming[r,u]`) but the negative branch uses
  `state.weights[r]` directly, with no cavity correction.

### Falsifiable test

Add `use_pull` alongside the existing `use_negative` flag and sweep the trust
factor `c` in `scripts/run_ablation.py`. Predictions, stated in advance:

- `dense` should improve. The pseudo-ranges are 1.32x better than a direct link
  there, and the push is retained rather than traded away.
- `test` should improve little. Pseudo-ranges are 0.84x a direct link, and the
  failure there is wrong azimuth (estimates are under-dispersed: mean `|est|`
  11.6 / 14.6 / 18.1 at hop 1/2/3 against a true 25.4), not wrong magnitude.
- `spread` should fall relative to RMSE — the overconfidence cost. If `spread`
  collapses while RMSE stays flat, `c` is too small.
- Large `c` must reproduce the current numbers exactly. That is the regression
  test for the implementation.
- The 4-arm grid (push on/off x pull on/off) at hop 2 should show push-only
  beating pull-only, since the ablation already gives push the larger effect.

Use >= 8 seeds. The current 2-seed spread on `dense` hop 2 is +-0.489, larger
than most of the effects above.

---

## 3. Open, smaller

- **NBP vs MDS on `dense` is unresolved.** 2 seeds give 2.976 +-0.489 against
  MDS 2.987 — parity, not a win. Needs ~8 seeds before any claim is made.
- **RMSE bottoms at iteration ~5 then drifts up** (2.49 -> 2.83 by iteration 10
  on dense). Particle depletion. Systematic resampling instead of multinomial in
  `particles.resample_indices` is a one-line change worth measuring.
- **Anchor placement matters more than anchor count.** `test` has 4 random
  anchors and 29 of 96 targets hear none; `dense` has 7 grid anchors and 90 of
  93 hear one. Worth an explicit sweep, since it is cheap and probably dominates
  every algorithmic effect measured so far.
