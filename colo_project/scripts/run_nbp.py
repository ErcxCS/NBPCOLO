"""Run NBP on a scenario against the MDS and CRLB baselines."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="test")
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "dataset" / "scenarios")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--n-particles", type=int, default=125)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--n-hop", type=int, default=2)
    parser.add_argument("--meters", type=float, default=100.0)
    parser.add_argument("--no-priors", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    from colo_project.dataset.data_loader import load_or_generate
    from colo_project.mds.classic_mds import ClassicMDS
    from colo_project.nbp.core import NBP, NBPConfig
    from colo_project.utils import io
    from colo_project.utils.metrics import (
        crlb, euclidean_metrics, per_node_peb, summarize_crlb,
    )
    from colo_project.utils.visualizations import plot_results

    sc = load_or_generate(args.scenario, args.scenarios_dir)
    t = sc.num_anchors

    cov = crlb(sc.B, sc.X_true, sc.num_anchors,
               alpha=sc.alpha, d0=sc.d0, sigma_db=sc.noise)
    crlb_rms = summarize_crlb(cov)["CRLB rms threshold"]
    peb = per_node_peb(cov)

    mds = ClassicMDS(dim=sc.dim)
    _, _, rigid = mds.run_mds(sc.X_true, sc.D, sc.full_D, sc.num_anchors)
    mds_rmse = euclidean_metrics(sc.targets, rigid[t:])[0]

    cfg = NBPConfig(
        n_particles=args.n_particles, n_iter=args.n_iter,
        n_batches=args.n_batches, radius=args.radius, n_hop=args.n_hop,
        meters=args.meters, use_priors=not args.no_priors, seed=args.seed,
    )

    print(f"scenario        : {sc.name} (seed {sc.seed})")
    print(f"nodes / anchors : {sc.n_nodes} / {sc.num_anchors}")
    print(f"mean degree     : {sc.mean_degree:.2f}")
    print(f"CRLB rms        : {crlb_rms:.3f}")
    print(f"MDS RMSE        : {mds_rmse:.3f}")
    print("NBP:")

    res = NBP(sc, cfg).run()

    print(f"NBP RMSE final  : {res.rmse[-1]:.3f}  "
          f"(best {res.rmse.min():.3f} at iter {res.rmse.argmin() + 1})")
    print(f"degenerate/empty: {res.n_degenerate} / {res.n_empty_bbox}")
    print(f"runtime         : {res.runtime_s:.1f}s")

    out = io.result_dir(args.results_dir, sc.name, sc.seed)
    io.save_json(out / "metrics_nbp.json", {
        "scenario": res.scenario,
        "seed": res.seed,
        "algorithm": "nbp",
        "git_sha": io.git_sha(),
        "config": res.config,
        "crlb_rms": crlb_rms,
        "mds_rmse": mds_rmse,
        "rmse": res.rmse,
        "mae": res.mae,
        "med": res.med,
        "spread": res.spread,
        "n_degenerate": res.n_degenerate,
        "n_empty_bbox": res.n_empty_bbox,
        "runtime_s": res.runtime_s,
    })
    io.save_arrays(
        out / "arrays_nbp.npz",
        estimates=res.estimates, estimates_hist=res.estimates_hist,
        particles=res.particles, weights=res.weights, peb=peb,
    )

    it = range(1, len(res.rmse) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(it, res.rmse, "o-", label="NBP RMSE")
    ax.plot(it, res.med, "s--", label="NBP median", alpha=0.7)
    ax.axhline(mds_rmse, color="tab:orange", ls=":", label="MDS RMSE")
    ax.axhline(crlb_rms, color="tab:red", ls="-.", label="CRLB")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error (m)")
    ax.set_title(f"{sc.name} seed {sc.seed}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=120)

    fig2 = plot_results(sc.X_true, np.vstack([sc.anchors, res.estimates]),
                        sc.num_anchors, show_lines=True, show_anchors=True,
                        title=f"NBP — RMSE {res.rmse[-1]:.2f}")
    fig2.savefig(out / "layout_nbp.png", dpi=120)

    if args.show:
        plt.show()
    plt.close("all")
    print(f"wrote           : {out}")


if __name__ == "__main__":
    main()
