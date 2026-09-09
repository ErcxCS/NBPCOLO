"""Real geotagged capture points as a position source for a scenario.

The datasets in `Image_location_datasets/` are lists of street-level captures,
each carrying `latitude`/`longitude` (the other keys -- bearing, pitch, the
image filename -- describe the photograph, not the node, and are ignored).

Two conversions stand between those and an `X_true` the rest of the project
accepts:

  * **Units.** `radius`, `d0` and every distance downstream are metres, so
    degrees have to be projected. Over a 1.3 km extent an equirectangular
    projection about the dataset centroid differs from the geodesic by a
    relative `(L/R)^2`, about 4e-8, so nothing more elaborate is warranted.
  * **Origin.** `NBPConfig.meters` builds the bbox limits as
    `[-meters/2, +meters/2]` per axis, so the field must be centred on zero.
    Real coordinates are not, and are not square either -- hence `meters` is
    returned rather than assumed, and must be passed to the run.

Anchors come first, as everywhere else in the project, so the loader also does
the reordering.

`geo_frame` returns the parameters of both conversions plus the reordering, so
that `unproject` can map an *estimate* in scenario metres back to latitude and
longitude. `load_geo_positions` is the thin generator-facing view of it; the
two share one code path so the forward and inverse maps cannot drift apart.
"""

import json
from pathlib import Path

import numpy as np

EARTH_R = 6371000.0


def _project(lat, lon):
    """Equirectangular metres about the centroid of the points themselves.

    Returns the reference latitude/longitude alongside, since inverting the
    projection needs them and nothing else records them.
    """
    lat0, lon0 = lat.mean(), lon.mean()
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y]), float(lat0), float(lon0)


def farthest_point_anchors(X, k):
    """Indices of `k` well-spread points, by farthest-point sampling.

    The CRLB is anchored, and anchors that huddle together leave the geometry
    nearly unresolved; spreading them over the hull is what keeps the FIM well
    conditioned. Deterministic -- it starts from the point farthest from the
    centroid -- so no `rng` is needed or taken.
    """
    if not 0 < k <= len(X):
        raise ValueError(f"need 0 < k <= {len(X)}, got {k}")
    centre = X.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(X - centre, axis=1)))]
    gap = np.linalg.norm(X - X[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(gap))
        chosen.append(nxt)
        gap = np.minimum(gap, np.linalg.norm(X - X[nxt], axis=1))
    return np.array(chosen)


def geo_frame(path: Path, num_anchors: int, pad: float = 0.05) -> dict:
    """Load one capture-point dataset with everything needed to invert it.

    Keys: `entries` (the raw JSON records, in file order), `X` (metres,
    centred, anchors first), `meters`, `order` (scenario row -> source index),
    `lat0`/`lon0` (projection reference) and `offset` (the bbox recentring
    that was subtracted). Deterministic, so calling it later reproduces the
    exact frame a scenario was generated in.
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    lat = np.array([e["latitude"] for e in entries], dtype=float)
    lon = np.array([e["longitude"] for e in entries], dtype=float)
    X, lat0, lon0 = _project(lat, lon)

    # Recentre on the bounding box rather than the centroid: a path-shaped
    # dataset has an off-centre centroid, which would waste half the field.
    offset = (X.min(axis=0) + X.max(axis=0)) / 2.0
    X = X - offset
    meters = float((1.0 + pad) * np.ptp(X, axis=0).max())

    anchors = farthest_point_anchors(X, num_anchors)
    rest = np.setdiff1d(np.arange(len(X)), anchors)
    order = np.concatenate([anchors, rest])
    return {"entries": entries, "X": X[order], "meters": meters,
            "order": order, "lat0": lat0, "lon0": lon0, "offset": offset}


def unproject(X, frame: dict):
    """Map `(n, 2)` scenario metres back to `(lat, lon)` degrees.

    The exact inverse of `geo_frame`'s recentring and projection, so a true
    position round-trips to the latitude and longitude it was read from, and
    an *estimate* lands where the estimator put it on the map.
    """
    X = np.asarray(X, dtype=float) + frame["offset"]
    lat = frame["lat0"] + np.degrees(X[:, 1] / EARTH_R)
    lon = frame["lon0"] + np.degrees(
        X[:, 0] / (EARTH_R * np.cos(np.radians(frame["lat0"]))))
    return lat, lon


def load_geo_positions(path: Path, num_anchors: int, pad: float = 0.05):
    """Load one capture-point dataset as `(X_true, meters, num_anchors)`.

    `X_true` is metres, centred on the origin, anchors first. `meters` is the
    side of the square field the caller must hand to `NBPConfig`, padded so
    that no node sits exactly on the bbox boundary.
    """
    frame = geo_frame(path, num_anchors, pad)
    return frame["X"], frame["meters"], num_anchors
