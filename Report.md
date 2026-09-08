# Real capture-point datasets as an NBP position source

Result of the plan in `TODO.md`: swap the *source* of `X_true` from synthetic to
real, keep every downstream stage untouched, and see what NBP does.

All four datasets in `dataset/Image_location_datasets/` load and run. **Two of
the four are localizable; two are not, and the CRLB says so before any algorithm
runs.** The dividing line is local geometry, not connectivity.

---

## 1. What the datasets actually are

Despite the `_transforms.json` suffix these are not NeRF camera transforms. Each
file is a flat JSON **list** of geotagged street-level captures:

```json
{"index": 0, "filename": "frame_0000.jpg", "id": "mvCWtJ61fDXqttVksYvFWA",
 "latitude": 41.37551748934333, "longitude": 2.177696596608505,
 "distance_m": 35.73, "bearing_deg": 10.96, "pitch_deg": 38.08, "date": "2024-08"}
```

Only `latitude`/`longitude` describe the *node*. The remaining keys describe the
photograph (where the camera was pointing, how far the subject was) and are
ignored. Two of the four files use `heading`/`pitch`/`roll` instead of
`bearing_deg`/`pitch_deg`/`distance_m`, so nothing may assume a fixed key set.

| dataset | n | extent (m) | nearest-neighbour med | closest pair |
|---|---|---|---|---|
| Amsterdam_Stopera_Road | 256 | 833 x 990 | 9.93 | 3.33 |
| Barcelona_Columbus_Orbit | 153 | 212 x 252 | 9.62 | 3.64 |
| DenHaag_Straight_Corridor | 205 | 1267 x 539 | 10.11 | 6.70 |
| Paris_ArcDeTriomphe_Orbit_Clean | 200 | 307 x 276 | 9.84 | 1.31 |

Captures sit ~10 m apart along a route in all four. No duplicate positions.

---

## 2. Method

`dataset/geo_positions.py`, wired into `generate_scenario` behind a `positions`
key in the scenario JSON. Three conversions, all anticipated by `TODO.md`:

**Projection.** Equirectangular metres about the dataset centroid. Over a 1.3 km
extent this differs from the geodesic by a relative `(L/R)^2`, about `4e-8`, so
nothing more elaborate is warranted.

**Origin and extent.** `NBPConfig.meters` builds the bbox limits as
`[-meters/2, +meters/2]` per axis, so the field must be centred on zero. Real
coordinates are not, and are not square. Recentring uses the **bounding box
centre, not the centroid** — a path-shaped dataset has an off-centre centroid,
which would waste half the field. `meters` is returned, never assumed.

**Anchors.** Farthest-point sampling, then reordered to the front per the
anchors-first convention. The CRLB is anchored, and anchors that huddle leave the
geometry nearly unresolved, so spreading them over the hull is what keeps the FIM
conditioned. Deterministic, so it takes no `rng`.

`placement` is forced to `false` when positions are loaded, otherwise the hex
grid would overwrite the real anchor coordinates.

**Radius** is chosen per dataset to hit mean degree ~20, matching the `dense`
scenario, which is the regime where NBP is known to be well-posed:

| dataset | radius (m) | meters | N | anchors | mean degree |
|---|---|---|---|---|---|
| amsterdam | 100 | 1039.0 | 256 | 8 | 20.4 |
| barcelona | 40 | 264.4 | 153 | 8 | 20.6 |
| denhaag | 100 | 1330.1 | 205 | 8 | 21.2 |
| paris | 50 | 322.8 | 200 | 8 | 22.1 |

Everything after `X_true` — path loss, shadowing, radius thresholding, the `.npz`
contract, `Scenario`, MDS, CRLB, NBP — ran unchanged, as `TODO.md` predicted.

---

## 3. Results

| dataset | PEB med | MDS rigid | NBP final | NBP med | runtime |
|---|---|---|---|---|---|
| barcelona | **1.40** | 10.59 | **10.63** | **5.68** | 100 s |
| paris | **2.34** | **11.04** | 42.07 | 10.70 | 140 s |
| denhaag | 61.0 | 164.64 | 40.03 | 13.65 | 101 s |
| amsterdam | 77.4 | 373.96 | 68.97 | 47.18 | 239 s |

Per-node error, NBP:

| dataset | med | p75 | p90 | p95 | max | > 50 m |
|---|---|---|---|---|---|---|
| barcelona | 5.68 | 8.09 | 9.39 | 14.72 | 57.2 | 2/145 |
| paris | 10.70 | 22.16 | 78.85 | 104.47 | 157.7 | 33/192 |
| denhaag | 13.65 | 43.65 | 74.85 | 94.50 | 101.9 | 41/197 |
| amsterdam | 47.18 | 69.53 | 104.83 | 156.17 | 203.0 | 113/248 |

### 3.1 Geometry decides feasibility, not connectivity

Local anisotropy is the SVD ratio `s0/s1` of each node's within-radius
neighbours — how collinear the neighbourhood is.

| dataset | local anisotropy med / p90 | range sigma med | PEB med | targets PEB > 10 m |
|---|---|---|---|---|
| `dense` (reference) | 1.3 / 1.9 | 1.50 m | 0.74 | — |
| barcelona | 1.4 / 2.0 | 1.94 m | 1.40 | 2/145 |
| paris | 1.3 / 1.9 | 2.49 m | 2.34 | 4/192 |
| denhaag | 6.8 / 52.3 | 3.82 m | 61.0 | 189/197 |
| amsterdam | 6.1 / 56.8 | 3.97 m | 77.4 | 243/248 |

The orbits are thick annuli with radial spokes — locally 2-D, indistinguishable
from the synthetic `dense` scenario on this measure.

The road and the corridor are **one node wide**: a vehicle driving a route. Every
node's neighbours lie along the same line, so the cross-track direction is
unconstrained and range-only trilateration is degenerate.

The decisive detail: **Amsterdam's worst-conditioned nodes have degree 19-20,
above its mean of 20.4** (PEB 953, 840, 786 m). More neighbours do not help when
all of them are collinear. This is not a connectivity problem and no algorithm
recovers from it.

### 3.2 NBP and MDS swap places on the path-shaped datasets

NBP beats MDS by 4-5x on exactly the two datasets where MDS should struggle.
Geodesic over true Euclidean distance, on the noiseless graph:

| dataset | hops to connect | mean | p90 | max |
|---|---|---|---|---|
| `dense` | 6 | 1.009 | 1.025 | 1.19 |
| barcelona | 8 | 1.027 | 1.068 | 2.22 |
| paris | 8 | 1.059 | 1.179 | 1.74 |
| denhaag | 19 | 1.183 | 1.398 | 2.40 |
| amsterdam | 22 | 1.353 | 2.002 | 3.03 |

This is the mechanism measured earlier for `dense`, pushed past breaking point.
There the inflation was ~1% and near-constant across distance bins, so classic
MDS absorbed it as a global scale — a similarity transform it reproduces exactly.
On a road the shortest path must follow every bend, so it overestimates the
straight line by 35% on average and 2x for the top decile. That is no longer a
smooth multiplicative bias, and MDS's distance completion collapses with it.

NBP only ever consumes direct one-hop ranges (`D_direct`; the n-hop matrix is a
reachability gate for the push term, not a distance source), so it degrades far
more gracefully. Both remain unusable on these two datasets, but for different
reasons: MDS fails at distance completion, NBP fails at the underlying geometry.

### 3.3 Paris: a coherent block in the wrong place

Paris is the one genuine anomaly — well-posed geometry (PEB med 2.34), MDS at
11.04, and NBP at 42.07, the only case where NBP loses to the baseline.

`figures/layout_nbp.png` shows why. Most of the ring is estimated well, but one
contiguous arc is displaced as a **coherent block** with roughly parallel error
lines — 33/192 nodes over 50 m out, max 157.7 m against a 307 m ring. Parallel,
not scattered: the segment satisfies its own internal ranges but sits in the
wrong place. A locally consistent, globally wrong configuration.

NBP's median of 10.70 is competitive with MDS's whole-set RMSE; the 42.07 is
entirely that block. A global spectral method does not have this failure mode,
which is exactly why MDS wins here. The natural remedy is the MDS warm-start
(`mds_init`) that `CLAUDE.md` records as deliberately not ported.

---

## 4. Caveats

- **`meters` and `radius` must be passed on the command line.** `Scenario` stores
  neither, and `run_nbp --meters` defaults to 100 against real fields of
  264-1330 m. Both are now written into the `.npz` (harmless — `Scenario.load`
  ignores unknown keys) but nothing reads them yet. Forget the flag and you
  silently get a 100 m bbox prior on a 1.3 km field.
- **`summarize_crlb` reports an rms over per-node PEBs, which is not robust.**
  Barcelona's CRLB rms of 113 m comes entirely from *one* degree-3 node with
  PEB 1364 m; 143 of its 145 targets are under 10 m and the median is 1.40.
  Judge these datasets by median PEB, never the rms.
- Anchors are capture points repurposed as anchors, not separately deployed
  infrastructure. On a path-shaped dataset they are necessarily strung along the
  same line as everything else.
- One seed (31), one noise level (1.0 dB), 10 iterations. No sweep.

---

## 5. Reproducing

```powershell
python -m colo_project.scripts.run_mds --scenario barcelona
python -m colo_project.scripts.run_nbp --scenario barcelona --radius 40  --meters 264.4
python -m colo_project.scripts.run_nbp --scenario paris     --radius 50  --meters 322.8
python -m colo_project.scripts.run_nbp --scenario denhaag   --radius 100 --meters 1330.1
python -m colo_project.scripts.run_nbp --scenario amsterdam --radius 100 --meters 1039.0
```

Scenario `.npz` files regenerate automatically from the configs in
`dataset/scenarios/`.

---

## 6. What this suggests next

1. **Carry `meters` and `radius` on `Scenario`** so the flags cannot be
   forgotten. The values are already in the `.npz`; this is a read.
2. **Port `mds_init`** and check whether an MDS warm-start removes the Paris
   block. It is the one failure here that is an algorithm problem rather than a
   geometry problem, so it is the only one worth attacking.
3. **Report median PEB alongside the rms** in `summarize_crlb`, or in place of
   it. The rms is dominated by whichever single node is worst conditioned.
4. Do not spend effort on amsterdam/denhaag as localization benchmarks. They are
   useful as *negative* controls — a demonstration that the CRLB predicts
   failure from geometry alone, before any estimator runs.
