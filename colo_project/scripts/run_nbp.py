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
    from colo_project.nbp.bbox import bbox_area
    from colo_project.nbp.core import NBP, NBPConfig
    from colo_project.utils import io
    from colo_project.utils.metrics import (
        crlb, euclidean_metrics, per_node_error, per_node_peb, summarize_crlb,
    )
    from colo_project.utils.visualizations import (
        plot_convergence, plot_error_cdf, plot_error_vs_degree, plot_messages,
        plot_particles, plot_results, scenario_figures,
    )

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
    ex = res.extras
    io.save_arrays(
        out / "arrays_nbp.npz",
        estimates=res.estimates, estimates_hist=res.estimates_hist,
        particles=res.particles, weights=res.weights, peb=peb,
        particles_hist=ex["particles_hist"], weights_hist=ex["weights_hist"],
        spread_hist=ex["spread_hist"], bboxes=ex["bboxes"],
    )

    # Scenario-level, so beside the run dirs rather than inside one.
    for stem, fig in scenario_figures(sc).items():
        io.save_fig(fig, out.parent / f"{stem}.png")

    figs = out / "figures"
    estimates = np.vstack([sc.anchors, res.estimates])
    err_nbp = per_node_error(sc.targets, res.estimates)
    err_mds = per_node_error(sc.targets, rigid[t:])
    degrees = sc.B.sum(axis=1)[t:]
    # Anchors have zero spread; padding keeps plot_results free of any
    # offset-by-num_anchors indexing.
    radii = np.concatenate([np.zeros(t), ex["spread_hist"][-1]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_convergence(
        {"RMSE": res.rmse, "MAE": res.mae, "median": res.med},
        baselines={"MDS RMSE": mds_rmse, "CRLB": crlb_rms}, ax=axes[0],
        title=f"{sc.name} seed {sc.seed}")
    plot_convergence({"belief spread": res.spread}, ax=axes[1],
                     ylabel="spread (m)", title="reported uncertainty")
    fig.tight_layout()
    io.save_fig(fig, figs / "convergence.png")

    io.save_fig(
        plot_results(sc.X_true, estimates, sc.num_anchors, show_lines=True,
                     show_anchors=True, radii=radii,
                     title=f"NBP — RMSE {res.rmse[-1]:.2f}"),
        figs / "layout_nbp.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_error_cdf({"NBP": err_nbp, "MDS rigid": err_mds}, bound=peb,
                   ax=axes[0], title="per-node error")
    plot_error_vs_degree({"NBP": err_nbp, "MDS rigid": err_mds}, degrees,
                         ax=axes[1], title="error vs connectivity")
    fig.tight_layout()
    io.save_fig(fig, figs / "compare_nbp_mds_crlb.png")

    # Chosen from the data, never hardcoded: the target NBP did worst on.
    worst = t + int(err_nbp.argmax())
    picks = np.unique(np.linspace(0, cfg.n_iter - 1, 6).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    for k, i in enumerate(picks):
        plot_particles(ex["particles_hist"][i], ex["weights_hist"][i],
                       sc.X_true, sc.num_anchors, [worst], ax=axes.flat[k],
                       title=f"iteration {i + 1}")
    for extra in axes.flat[len(picks):]:
        extra.set_visible(False)
    fig.suptitle(f"belief of node {worst} (worst error, "
                 f"{err_nbp.max():.2f} m)")
    fig.tight_layout()
    io.save_fig(fig, figs / "particles_iter.png")

    widest = t + np.argsort(bbox_area(ex["bboxes"])[t:])[-4:]
    io.save_fig(
        plot_particles(ex["particles_hist"][0], ex["weights_hist"][0],
                       sc.X_true, sc.num_anchors, widest.tolist(),
                       bboxes=ex["bboxes"],
                       title="widest anchor-derived priors, iteration 1"),
        figs / "priors.png")

    io.save_fig(
        plot_messages(ex["proposals"], worst, res.particles, res.weights,
                      sc.X_true, sc.num_anchors, sc.D,
                      title=f"messages into node {worst}, "
                            f"iteration {cfg.n_iter}"),
        figs / "messages_node.png")

    if args.show:
        plt.show()
    plt.close("all")
    print(f"wrote           : {out}")


if __name__ == "__main__":
    main()
