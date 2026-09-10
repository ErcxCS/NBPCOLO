"""Export a run's estimates as geographic coordinates, joined to the source.

For a scenario whose positions came from a real capture-point dataset, this
pairs each node with the record it was built from and with what NBP and MDS
estimated for it, mapping the estimates out of scenario metres and back onto
the map through the same frame `geo_positions` projected them in.

The scenario must have anchors. They are what ties the estimate to an absolute
frame, and without that frame there is nothing to unproject through.

Position is the only thing estimated. `heading`/`pitch`/`roll` (and the orbit
files' `bearing_deg`/`pitch_deg`/`distance_m`) describe the camera, and no
stage of this project models orientation, so they are copied through under
`source` and never appear under `nbp` or `mds`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from colo_project.dataset.data_loader import load_or_generate
from colo_project.dataset.geo_positions import EARTH_R, geo_frame, unproject
from colo_project.utils import io
from colo_project.utils.metrics import per_node_error


def _latest(parent: Path, filename: str) -> Path:
    """The newest `run_NNNN` under `parent` that holds `filename`."""
    runs = sorted(p for p in parent.glob("run_*")
                  if (p / filename).exists())
    if not runs:
        raise FileNotFoundError(f"no run under {parent} holds {filename}")
    return runs[-1]


def export(name: str, scenarios_dir: Path, results_dir: Path) -> Path:
    cfg = json.loads((scenarios_dir / f"{name}.json").read_text())
    if "positions" not in cfg:
        raise ValueError(f"{name} has no `positions` key: not a real dataset")

    sc = load_or_generate(name, scenarios_dir)
    if sc.num_anchors == 0:
        # Anchor-free, the estimate is defined only up to translation,
        # rotation and reflection, so there is no frame to unproject it
        # through: the lat/lon it produced would be an artefact of an
        # arbitrary gauge. run_nbp scores such a run by Procrustes onto the
        # truth, but aligning with the truth and then calling the result a
        # geographic estimate is circular -- it is the answer being exported.
        raise ValueError(
            f"{name} has no anchors: an anchor-free estimate carries no "
            f"absolute frame, so it cannot be mapped back to latitude and "
            f"longitude. Export the anchored variant of this dataset.")

    source = (scenarios_dir / cfg["positions"]).resolve()
    frame = geo_frame(source, sc.num_anchors)
    if not np.array_equal(frame["X"], sc.X_true):
        raise RuntimeError(f"{name}: frame does not reproduce the scenario")

    parent = results_dir / f"{name}_seed{sc.seed}"
    nbp_run, mds_run = _latest(parent, "arrays_nbp.npz"), _latest(
        parent, "arrays_mds.npz")
    nbp = np.load(nbp_run / "arrays_nbp.npz")
    mds = np.load(mds_run / "arrays_mds.npz")

    t = sc.num_anchors
    # NBP estimates targets only; an anchor's belief is a copy of its known
    # position, so that is what it "estimated" for the anchor rows.
    x_nbp = np.vstack([sc.anchors, nbp["estimates"]])
    x_mds = mds["rigid"]
    err_nbp = per_node_error(sc.X_true, x_nbp)
    err_mds = per_node_error(sc.X_true, x_mds)
    spread = nbp["spread_hist"][-1]
    peb = nbp["peb"]
    degree = sc.B.sum(axis=1)

    lat_true, lon_true = unproject(sc.X_true, frame)
    lat_nbp, lon_nbp = unproject(x_nbp, frame)
    lat_mds, lon_mds = unproject(x_mds, frame)

    nodes = []
    for i in range(sc.n_nodes):
        src = int(frame["order"][i])
        is_anchor = i < t
        nodes.append({
            "node": i,
            "role": "anchor" if is_anchor else "target",
            "source_index": src,
            "degree": int(degree[i]),
            "peb_m": None if is_anchor else float(peb[i - t]),
            "source": frame["entries"][src],
            "true": {"lat": float(lat_true[i]), "lon": float(lon_true[i]),
                     "x_m": float(sc.X_true[i, 0]),
                     "y_m": float(sc.X_true[i, 1])},
            "nbp": {"lat": float(lat_nbp[i]), "lon": float(lon_nbp[i]),
                    "x_m": float(x_nbp[i, 0]), "y_m": float(x_nbp[i, 1]),
                    "error_m": float(err_nbp[i]),
                    "spread_m": None if is_anchor else float(spread[i - t])},
            "mds": {"lat": float(lat_mds[i]), "lon": float(lon_mds[i]),
                    "x_m": float(x_mds[i, 0]), "y_m": float(x_mds[i, 1]),
                    "error_m": float(err_mds[i])},
        })

    out = parent / "estimates_geo.json"
    io.save_json(out, {
        "dataset": name,
        "source_file": source.name,
        "git_sha": io.git_sha(),
        "estimated": ["latitude", "longitude"],
        "not_estimated": (
            "heading/pitch/roll and bearing_deg/pitch_deg/distance_m are "
            "camera metadata copied from the source file. This project "
            "estimates 2-D position from range measurements only; no stage "
            "models orientation, so there is no estimate of them."),
        "scenario": dict(cfg, n_nodes=sc.n_nodes, num_anchors=sc.num_anchors,
                         mean_degree=sc.mean_degree, alpha=sc.alpha,
                         d0=sc.d0, meters=frame["meters"]),
        "projection": {
            "type": "equirectangular about the source centroid, then "
                    "recentred on the bounding-box centre",
            "earth_radius_m": EARTH_R,
            "lat0": frame["lat0"], "lon0": frame["lon0"],
            "offset_m": frame["offset"],
            "inverse": "lat = lat0 + degrees((y_m + offset_y) / R); "
                       "lon = lon0 + degrees((x_m + offset_x) "
                       "/ (R cos lat0))",
        },
        "runs": {"nbp": nbp_run.name, "mds": mds_run.name},
        "metrics": {
            "nbp": io.load_json(nbp_run / "metrics_nbp.json"),
            "mds": io.load_json(mds_run / "metrics.json"),
        },
        "nodes": nodes,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+",
                        default=["amsterdam", "barcelona", "denhaag", "paris"])
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "dataset" / "scenarios")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results")
    args = parser.parse_args()

    for name in args.scenarios:
        out = export(name, args.scenarios_dir, args.results_dir)
        print(f"{name:<10} -> {out}")


if __name__ == "__main__":
    main()
