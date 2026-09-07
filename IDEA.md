# Research ideas

Algorithm changes, not refactoring. Nothing here is implemented. Each entry
records the evidence for and against so it does not have to be re-derived.

Measurements below are on `dense` (100 nodes, 7 grid anchors, R=30, sigma 1 dB)
and `test` (100 nodes, 4 random anchors, R=20), both seed 31, at commit 51411f1.

---

## 1. Two-sided multi-hop constraint: `R < d <= k*R`

**Status:** plausible, untested. Proposed by Berk, 2026-09-07.

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
msg(x_u) = E_r[ (1 - detect(d, R)) * ceil(d, k*R) ]        d = ||x_u - p_r||
```

with `ceil` a soft upper gate rather than a hard cutoff, since ~1% of pairs
genuinely violate it. Slack should widen with k. Cost is near zero:
`negative_information` already forms the full pairwise `diff_sq`, so the ceiling
is one more elementwise term on a matrix that is already being built.

### Objections

- **Double counting.** The hop count k is a deterministic function of the same
  connectivity the one-hop messages already encode, so this is not independent
  evidence. Loopy BP will get more overconfident. Expect `spread` to tighten
  faster than RMSE improves — watch both, not just RMSE. This is the strongest
  argument against, and it is probably why the original authors used only the
  binary non-detection, which *is* a genuinely new observation.
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

Add `use_ceiling` alongside the existing `use_negative` flag and re-run
`scripts/run_ablation.py`. Predictions, stated in advance:

- `dense` hop 2: small or no improvement; it is already at MDS parity and the
  ceiling binds on only ~4% of estimates.
- `dense` hop 3-4: should stop the degradation (2.98 -> 3.69 -> 3.73 in final
  iterate) if the loose-annuli argument is right.
- `test`: little change expected, for the reason above. If it *does* improve
  substantially, the azimuth diagnosis is wrong and worth revisiting.
- `spread` should fall relative to RMSE in all arms (the overconfidence cost).

Use >= 8 seeds. The current 2-seed spread on `dense` hop 2 is +-0.489, larger
than most of the effects above.

---

## 2. Multi-hop pseudo-ranges as soft positive messages

**Status:** speculative.

Stronger version of (1): instead of an inequality, use `D_hop[u,r]` as an actual
range for multi-hop pairs, giving a proper annulus message. This is what MDS
consumes — it escalates the hop count until the graph is connected and treats
every pseudo-range as a distance — and MDS beats NBP 5.10 vs ~34 on `test`.

Blocked on the measurement in (1): `D_hop` is downward-biased by min-over-paths
(-0.65 m mean on dense one-hop entries, worse at higher hop counts), so it would
need bias correction first. The bias grows with graph density and hop count, so
a correction would have to be a function of both.

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
