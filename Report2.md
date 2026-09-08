# Anchor-free COLO: why NBP collapses, and what the push term is actually worth

Follow-up to `Report.md`. Same four real capture-point datasets, same seed (31),
same noise, same radii — but with **zero anchors**. The question was what NBP
does with no absolute reference. The answer turned out to be less about anchors
than about *initialisation*, and it put a number on the negative-information
("push") term that the docstrings had not.

**Headline: NBP's push/pull machinery is correct and does what it claims. It is
a local refiner with no mechanism for discovering global structure, the push is
a ~2% effect on the resample, and the anchor-free collapse is created by the
*pull* on iteration 1 — before the push ever gets a turn.**

---

## 1. Scoring without a frame

Range-only measurements are invariant to translation, rotation and reflection,
so with no anchors an estimate is only defined up to a rigid map. Two things had
to change:

**The bound.** `metrics.crlb` refuses `num_anchors == 0` by design — the FIM is
singular. `metrics.crlb_anchor_free` restricts it instead of regularising it:
`gauge_basis` writes the nullspace down analytically (two translations and the
rotation generator at `X_true`), `Q` spans the orthogonal complement, and
`Q.T @ fim @ Q` is full rank and inverts exactly. There is no Tikhonov `eps` and
no rank tolerance — the deficiency is known to be 3, not discovered from a
threshold, which is what keeps this a statement about geometry.

Validated on `dense`: the FIM annihilates the basis to `4e-15` relative, the
singular values gap by `6e10` exactly at `2N-3`, and the free PEB median comes
out at **0.897** against the anchored **0.742** — looser, as dropping
information must be.

**The error.** `metrics.rigid_transform` / `align_rigid` fit the best
rotation/reflection + translation (no scaling — ranges fix the scale) onto the
truth. `run_nbp` reports raw *and* aligned when `num_anchors == 0`, and maps
ground truth back into NBP's own frame for the particle/message figures rather
than leaving them misaligned.

---

## 2. Results

| dataset | free PEB med | MDS 8-anchor | MDS 0-anchor | NBP 8-anchor | NBP 0-anchor (aligned) | NBP med |
|---|---|---|---|---|---|---|
| barcelona | 13.45 | 10.59 | **9.37** | 10.63 | 75.94 | 56.7 |
| paris | 3.42 | 11.04 | **10.16** | 42.07 | 109.90 | 99.8 |
| denhaag | 3644.7 | 164.64 | **153.04** | 40.03 | 442.38 | 470.2 |
| amsterdam | 9522.6 | 373.96 | **363.42** | 68.97 | 410.64 | 367.8 |

**MDS is unaffected by removing the anchors** — it moves by under 6%, and
*improves*. In hindsight that is obvious: classic MDS never used anchors to
build the embedding, only to register it. Score with a free Procrustes over all
N nodes and you measure exactly what it always computed, fit slightly better
than an 8-row registration could manage.

**NBP degrades 7x, 2.6x, 11x and 6x.** On denhaag it goes from beating MDS 4x to
losing to it 3x. Every ranking in `Report.md` reverses.

### 2.1 The failure is shape, not gauge

| dataset | raw | aligned | recovered by alignment |
|---|---|---|---|
| barcelona | 94.6 | 75.9 | 20% |
| paris | 118.5 | 109.9 | 7% |
| denhaag | 443.1 | 442.4 | 0.2% |
| amsterdam | 435.5 | 410.6 | 6% |

If NBP had recovered the network and merely lost the frame, the aligned numbers
would land near the 8-anchor ones. They do not. The arbitrary frame accounts for
under a fifth of the error everywhere and essentially none of it on denhaag.

### 2.2 It is a collapse

Estimated extent as a fraction of true, per iteration:

| dataset | iter 1 | iter 5 | iter 10 |
|---|---|---|---|
| barcelona | 0.18 | 0.44 | 0.49 |
| paris | 0.19 | 0.40 | 0.44 |
| denhaag | 0.17 | 0.15 | 0.17 |
| amsterdam | 0.16 | 0.18 | 0.20 |

The network implodes to a sixth of its true size on the first iteration and
never recovers. The two that partially re-expand (barcelona, paris) are the
rigid ones; the two that stall are the near-flexible ones — see 4.1.

NBP also reports a belief spread of 2.5-3.5 m against 76-440 m of error, and
RMSE now degrades monotonically from iteration 1 rather than bottoming at 5 as
it does with anchors.

### 2.3 Two different failures, separated by range residuals

Median `|predicted - measured|` over measured edges:

| dataset | truth | NBP | MDS |
|---|---|---|---|
| barcelona | 1.14 | 3.73 | 2.44 |
| paris | 1.43 | 4.80 | 4.19 |
| denhaag | 2.08 | 4.21 | **6.73** |
| amsterdam | 2.20 | 4.63 | **6.97** |

On barcelona/paris NBP fits the *measurements* worse than MDS does, so it did
not find a legitimate alternative embedding — it simply failed. On
denhaag/amsterdam it fits them better while being 3x more wrong, which is the
locally-consistent-globally-wrong signature the near-flex modes permit.

---

## 3. What the push term is actually worth

`potentials.py` claims negative information "is what makes multi-hop NBP better
than one-hop NBP". Measured, that is an overstatement.

`log_message_product` rescales its output so the maximum is 1, so what steers
the resample is the **spread** of the log-message sum across the candidate pool;
a factor common to every particle is exactly invisible. Measuring that spread:

| | pull | push | push share |
|---|---|---|---|
| barcelona, 0 anchors, iter 1 | 17.5 | 0.27 | **1.5%** |
| barcelona, 0 anchors, iter 2 | 63.9 | 15.2 | 19.3% |
| barcelona, **8 anchors**, iter 1 | 258.8 | 4.6 | **1.8%** |
| barcelona, **8 anchors**, iter 3 | 296.6 | 5.9 | 2.0% |

Note the 8-anchor rows: the push was never carrying the run there either. This
is structural rather than a bug. A pull message is a KDE on a range annulus of
width `sigma ~ 2 m`, so its log varies by hundreds across a 250 m pool. A push
message is `1 - E[detect]`, bounded in `[0, 1]`, so its log varies by ~0.2. The
pull is a sharp likelihood and the push is a soft bounded one: a ~100x
asymmetry, by construction.

End-to-end on barcelona anchor-free: push ON **75.94**, push OFF **78.01**,
`n_hop=1` **78.01**. Removing the push costs 2.6%.

(`n_hop=1` and push-off agreeing to the digit is a correctness check that
passes: with `use_negative=False`, `n_hop` only affects range smoothing, and
`NBP.__init__` restores the direct reading over one-hop pairs, so the two are
the same computation.)

### 3.1 More hops helps — and shows why more hops is the wrong lever

| n_hop | push set (mean) | % of network | median pushed pair | that message's value | aligned RMSE | extent/true | runtime |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0% | — | — | 78.01 | 0.29 | — |
| 2 | 32.7 | 21.5% | 59 m | 0.659 | 75.94 | 0.49 | 158 s |
| 3 | 69.2 | 45.5% | 77 m | 0.839 | 72.62 | 0.67 | 162 s |
| 5 | 118.4 | **77.9%** | 99 m | **0.953** | 56.92 | 0.82 | 242 s |
| MDS | — | — | — | — | **9.36** | — | ~5 s |

Raising `n_hop` does improve things monotonically, and the extent column shows
the push doing exactly what it is supposed to: forcing the network back apart,
0.29 -> 0.82 of true scale.

But `detection_prob` is a Gaussian falloff `exp(-d^2 / 2r^2)`, not a hard
cutoff, so the pairs added at higher hop counts are the *least* informative
ones. At `n_hop=5` the median push message is 0.953 — "you are probably not
within 40 m of a node 99 m away", which was never in doubt. Diminishing returns
are baked into the potential, and the cost is O(N^2).

At `n_hop=5` the push set is 78% of the network. That is no longer a Markov
random field in any useful sense — it is a dense all-pairs repulsion, a
mediocre spring layout — and it is still 6x worse than 5 seconds of MDS.

---

## 4. The actual cause: the pull collapses the network on iteration 1

Phase A, `nbp/core.py`:

```python
detect = detection_prob(X_ru, mu[u], cfg.radius)
```

`mu[u]` is u's **weighted belief mean**. With no anchors, u's prior is uniform
over the whole field, so `mu[u]` is the field centre — for every node at once:

| | init `mu[u]` std across nodes | true position std | iter-1 message centres, std |
|---|---|---|---|
| 0 anchors | **(6.9, 6.4)** m | (49.9, 62.1) m | **(7.0, 6.8)** m |
| 8 anchors | (49.1, 54.6) m | (49.4, 61.5) m | (46.6, 50.3) m |

All 153 nodes believe they are at the origin, so all ~3000 first-iteration
messages are blobs centred on `(-0.5, 0.6)`. The product of 21 co-centred blobs
of width 78 m is a tight peak at the origin. The collapse is complete at
iteration 1 — extent 0.18 — before the push has contributed anything.

The push cannot undo it afterwards, and not because it is too weak in
principle. It is a **reweighting of a pool that only the pull generates**: it
can never create mass where there are no particles. And once everything is
collapsed, every candidate is penalised equally, so after the rescale-by-max in
`log_message_product` there is no gradient left to follow at all.

### 4.1 Caveat: the anchor-free bound is only trustworthy on paris

Eigenvalues of the reduced FIM reveal extra near-zero modes beyond the three
gauge directions. Those are genuine flex — deformations the ranges cannot see:

| dataset | cond(QtFQ) | gap at 4th eigenvalue | free PEB med |
|---|---|---|---|
| paris | 1.3e7 | 2.0 — rigid, bound is real | 3.42 |
| barcelona | 7.4e8 | 13423 — one flex mode | 13.45 |
| denhaag | 8.6e10 | 55 | 3644.7 |
| amsterdam | 2.5e13 | 188 | 9522.6 |

denhaag's 3645 m and amsterdam's 9523 m are near-singularity artifacts, not
usable floors. MDS "beating" denhaag's bound is therefore not a contradiction:
the CRLB bounds unbiased estimators, and rank-2-projected MDS is biased.
Barcelona's 1.40 -> 13.45 jump is one real flex mode appearing when the anchors
go.

---

## 5. The fix is initialisation, not hop count

`NBPConfig.warm_start` seeds the particles from a global layout (MDS by
default) instead of from the anchor boxes or the whole field, with
`warm_halfwidth` as the seed box half-width (`None` -> `radius / 2`). Still
zero anchors, still scored by free Procrustes, same measurements. Aligned RMSE:

| dataset | MDS | NBP cold | NBP warm | warm best | warm med | extent/true | seed box |
|---|---|---|---|---|---|---|---|
| barcelona | 9.37 | 75.94 | **5.33** | 4.22 (it 3) | 4.64 | 1.05 | ±20 m |
| paris | 10.16 | 109.90 | 10.56 | 10.03 (it 3) | 9.99 | 1.08 | ±25 m |
| denhaag | 153.04 | 442.38 | **119.59** | 119.37 (it 1) | 119.34 | 1.09 | ±50 m |
| amsterdam | 363.42 | 410.64 | **271.71** | 266.66 (it 1) | 256.47 | 1.14 | ±50 m |

Warm start improves NBP on all four — 14x, 10x, 3.7x, 1.5x — and **the collapse
is gone**: extent goes from 0.17-0.49 of true to 1.05-1.14. What is left is a
slight over-expansion, not an implosion.

NBP-warm beats MDS on three of the four and ties on paris. That is the real
result: NBP is a *refiner*. Given a global method's output it improves on it;
asked to find global structure from a uniform prior it cannot, at any hop count.

Two details worth keeping:

**Paris.** With 8 anchors NBP scored 42.07 because one contiguous arc was
displaced as a coherent block (`Report.md` §3.3). Warm-started and anchor-free
it scores 10.56 with a p90 of 13.99 — the block is gone. That is exactly what
`Report.md`'s next-step #2 predicted a warm start would fix.

**Where the best iteration falls.** barcelona and paris peak at iteration 3, so
NBP genuinely refines the seed there. denhaag and amsterdam peak at iteration 1
and drift up afterwards — NBP cannot improve on MDS on a one-node-wide path, it
only holds station briefly. Both remain unusable in absolute terms (119.6 m and
271.7 m). Warm start fixes the initialisation problem; it does not touch the
flex problem in §4.1.

### 5.1 Warm start is not a free improvement

`dense` (7 anchors, the well-posed synthetic reference) is the counter-example:

| dense | best | final |
|---|---|---|
| cold | **2.808** (it 5) | **3.148** |
| MDS-warm | 2.911 (it 2) | 3.480 |

Warm start is slightly *worse* there. MDS scores 2.99 on `dense`, no better
than the 2.81 NBP reaches on its own, and the anchor bboxes were already
informative — so seeding from MDS only pins NBP to MDS's neighbourhood. Warm
start is a remedy for degenerate initialisation, not a default.

---

## 6. Caveats

- **Raw RMSE is meaningless on a warm anchor-free run by construction.** The
  seed is the raw MDS embedding, which double-centring leaves at the origin
  under an arbitrary rotation, so NBP inherits that frame: raw scores are 94.2,
  150.8, 697.7 and 294.9 against aligned 5.33, 10.56, 119.59, 271.71. This is
  why `run_nbp` reports both and why the seed must never be the truth-aligned
  layout — see the `mds_seed` capture in `run_nbp`.
- barcelona's 4.22 sits *below* its anchor-free PEB median of 13.45. Do not
  over-read that: barcelona has one near-flex mode inflating per-node PEBs, and
  a warm-started particle filter is biased, so it can sit under an unbiased
  bound in directions it never excites.
- The push measurements in section 3 are barcelona, iterations 1-3. The
  `potentials.py` docstring claim deserves a proper anchored ablation on
  `dense`/`test` before it is rewritten; `scripts/run_ablation.py` exists for
  exactly this but currently calls `crlb`, so it cannot run 0-anchor scenarios.
- `warm_halfwidth` defaults to `radius / 2` and was not tuned. The probe that
  found this used ±15 m on barcelona and scored 4.13 against the shipped
  default's 4.22 at ±20 m, so the result is not knife-edge — but no sweep was
  run.
- The `*_noanchor` scenarios reorder no nodes (there are no anchors to move to
  the front), so their noise draws land on different pairs than the anchored
  versions. Mean degrees match to within 0.5; the comparison is fair but not
  paired.
- One seed (31), one noise level (1.0 dB), 10 iterations, `P=125`. No sweep.

---

## 7. Reproducing

Cold, anchor-free (section 2):

```powershell
python -m colo_project.scripts.run_nbp --scenario barcelona_noanchor --radius 40  --meters 264.3933
python -m colo_project.scripts.run_nbp --scenario paris_noanchor     --radius 50  --meters 322.7597
python -m colo_project.scripts.run_nbp --scenario denhaag_noanchor   --radius 100 --meters 1330.0535
python -m colo_project.scripts.run_nbp --scenario amsterdam_noanchor --radius 100 --meters 1039.0495
```

Add `--warm-start` to any of those for section 5; `--n-hop 5` reproduces the
sweep row in 3.1, and `--warm-halfwidth` overrides the `radius / 2` default:

```powershell
python -m colo_project.scripts.run_nbp --scenario barcelona_noanchor --radius 40 --meters 264.3933 --n-hop 5
python -m colo_project.scripts.run_nbp --scenario barcelona_noanchor --radius 40 --meters 264.3933 --warm-start
python -m colo_project.scripts.run_nbp --scenario dense --radius 30 --warm-start
```

---

## 8. What this suggests next

1. **Rewrite the `potentials.py` claim** once an anchored ablation confirms the
   ~2% figure on `dense`. The push is real and it does hold scale, but "what
   makes multi-hop NBP better than one-hop NBP" is not supported at `n_hop=2`.
2. **Do not raise `n_hop` past 2 as a fix.** It buys RMSE at O(N^2) cost by
   adding near-vacuous constraints, and by `n_hop=5` the model is no longer
   local.
3. **`run_ablation.py` cannot run anchor-free scenarios** — it calls `crlb`
   directly. One branch, the same one `run_nbp` now has. With that fixed it
   could sweep `warm_halfwidth` and settle caveat 4 above.
4. The collapse mechanism in section 4 suggests a cheaper mitigation than a
   full warm start: on iteration 1 only, skip the `detect` reweighting when the
   receiver has no informative prior. Untested, and would matter for a
   deployment with no MDS stage.
5. **Warm-start NBP anchor-free now beats 8-anchor NBP on barcelona** (5.33 vs
   10.63) **and on paris** (10.56 vs 42.07). Re-running the anchored datasets
   with `--warm-start` is the obvious next measurement.
