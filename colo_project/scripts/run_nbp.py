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
    parser.add_argument("--warm-start", action="store_true",
                        help="seed particles from the MDS layout instead of "
                             "the anchor boxes / whole field")
    parser.add_argument("--warm-halfwidth", type=float, default=None,
                        help="seed box half-width in m (default: radius/2)")
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
        align_rigid, crlb, crlb_anchor_free, euclidean_metrics, per_node_error,
        per_node_peb, rigid_transform, summarize_crlb,
    )
    from colo_project.utils.visualizations import (
        plot_convergence, plot_error_cdf, plot_error_vs_degree, plot_messages,
        plot_particles, plot_results, scenario_figures,
    )

    sc = load_or_generate(args.scenario, args.scenarios_dir)
    t = sc.num_anchors
    # With no anchors there is no absolute frame: the estimate is defined only
    # up to translation, rotation and reflection. Both the bound and the
    # scoring then have to be gauge-invariant, so the whole run branches here
    # and nowhere else.
    free = sc.num_anchors == 0

    if free:
        cov = crlb_anchor_free(sc.B, sc.X_true,
                               alpha=sc.alpha, d0=sc.d0, sigma_db=sc.noise)
    else:
        cov = crlb(sc.B, sc.X_true, sc.num_anchors,
                   alpha=sc.alpha, d0=sc.d0, sigma_db=sc.noise)
    crlb_rms = summarize_crlb(cov)["CRLB rms threshold"]
    peb = per_node_peb(cov)

    mds = ClassicMDS(dim=sc.dim)
    x_hat, _, rigid = mds.run_mds(sc.X_true, sc.D, sc.full_D, sc.num_anchors)
    # Captured before the fallback below: `rigid` is about to be replaced by a
    # *truth-aligned* layout for scoring, and feeding that to the warm start
    # would hand the estimator the frame it is supposed to recover.
    mds_seed = x_hat if rigid is None else rigid
    if rigid is None:
        # run_mds has no anchors to register against and says so by returning
        # None; align on the whole point set instead. MDS keeps no absolute
        # frame at all -- double-centering puts its embedding at the origin --
        # so there is no raw score to report for it, only this one.
        rigid = align_rigid(x_hat, sc.X_true)
    mds_rmse = euclidean_metrics(sc.targets, rigid[t:])[0]

    cfg = NBPConfig(
        n_particles=args.n_particles, n_iter=args.n_iter,
        n_batches=args.n_batches, radius=args.radius, n_hop=args.n_hop,
        meters=args.meters, use_priors=not args.no_priors, seed=args.seed,
        warm_start=args.warm_start, warm_halfwidth=args.warm_halfwidth,
    )
    nbp = NBP(sc, cfg, init_positions=mds_seed)

    print(f"scenario        : {sc.name} (seed {sc.seed})")
    print(f"nodes / anchors : {sc.n_nodes} / {sc.num_anchors}")
    print(f"mean degree     : {sc.mean_degree:.2f}")
    gauge = "  (anchor-free, gauge-projected)" if free else ""
    print(f"CRLB rms        : {crlb_rms:.3f}{gauge}")
    # The rms is an rms over per-node PEBs and one ill-conditioned node runs
    # away with it; the median is what these datasets should be judged on.
    print(f"CRLB PEB median : {np.median(peb):.3f}")
    print(f"MDS RMSE        : {mds_rmse:.3f}"
          f"{'  (Procrustes-aligned)' if free else ''}")
    if cfg.warm_start:
        init_desc = f"MDS warm start, +/-{nbp.warm_halfwidth:.1f} m box"
    elif cfg.use_priors and t:
        init_desc = "anchor bboxes"
    else:
        init_desc = "whole field (uniform)"
    print(f"init            : {init_desc}")
    print("NBP:")

    res = nbp.run()

    # Anchor-free, the raw RMSE is mostly the arbitrary frame -- NBP holds one
    # only through its uniform-over-the-field initial prior, and nothing pins
    # the rotation at all. Score the shape as well, per iteration, so the two
    # can be told apart.
    rmse_aligned = med_aligned = aligned_hist = None
    if free:
        aligned_hist = np.array(
            [align_rigid(e, sc.targets) for e in res.estimates_hist]
        )
        scores = np.array(
            [euclidean_metrics(sc.targets, e) for e in aligned_hist]
        )
        rmse_aligned, med_aligned = scores[:, 0], scores[:, 2]

    print(f"NBP RMSE final  : {res.rmse[-1]:.3f}  "
          f"(best {res.rmse.min():.3f} at iter {res.rmse.argmin() + 1})"
          f"{'  [raw, frame included]' if free else ''}")
    if free:
        print(f"NBP aligned     : {rmse_aligned[-1]:.3f}  "
              f"(best {rmse_aligned.min():.3f} at iter "
              f"{rmse_aligned.argmin() + 1}), median "
              f"{med_aligned[-1]:.3f}")
    print(f"degenerate/empty: {res.n_degenerate} / {res.n_empty_bbox}")
    print(f"runtime         : {res.runtime_s:.1f}s")

    out = io.result_dir(args.results_dir, sc.name, sc.seed)
    io.save_json(out / "metrics_nbp.json", {
        "scenario": res.scenario,
        "seed": res.seed,
        "algorithm": "nbp",
        "git_sha": io.git_sha(),
        "config": res.config,
        "anchor_free": free,
        "warm_halfwidth_used": nbp.warm_halfwidth if cfg.warm_start else None,
        "crlb_rms": crlb_rms,
        "crlb_peb_med": float(np.median(peb)),
        "mds_rmse": mds_rmse,
        "rmse": res.rmse,
        "rmse_aligned": rmse_aligned,
        "med_aligned": med_aligned,
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
    # Score, and draw the layout, in the frame the truth lives in.
    est_scored = aligned_hist[-1] if free else res.estimates
    estimates = np.vstack([sc.anchors, est_scored])
    err_nbp = per_node_error(sc.targets, est_scored)
    err_mds = per_node_error(sc.targets, rigid[t:])
    degrees = sc.B.sum(axis=1)[t:]
    # Anchors have zero spread; padding keeps plot_results free of any
    # offset-by-num_anchors indexing.
    radii = np.concatenate([np.zeros(t), ex["spread_hist"][-1]])

    curves = {"RMSE": res.rmse, "MAE": res.mae, "median": res.med}
    if free:
        curves = {"RMSE raw": res.rmse, "RMSE aligned": rmse_aligned,
                  "median aligned": med_aligned}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_convergence(
        curves,
        baselines={"MDS RMSE": mds_rmse, "CRLB": crlb_rms}, ax=axes[0],
        title=f"{sc.name} seed {sc.seed}"
              f"{' — no anchors' if free else ''}")
    plot_convergence({"belief spread": res.spread}, ax=axes[1],
                     ylabel="spread (m)", title="reported uncertainty")
    fig.tight_layout()
    io.save_fig(fig, figs / "convergence.png")

    io.save_fig(
        plot_results(sc.X_true, estimates, sc.num_anchors, show_lines=True,
                     show_anchors=not free, radii=radii,
                     title=f"NBP — RMSE "
                           f"{(rmse_aligned if free else res.rmse)[-1]:.2f}"
                           f"{' (Procrustes-aligned)' if free else ''}"),
        figs / "layout_nbp.png")
    if free:
        # The same estimate before alignment: the gap between the two figures
        # is the gauge, and is not an estimation error.
        io.save_fig(
            plot_results(sc.X_true, np.vstack([sc.anchors, res.estimates]),
                         0, show_lines=True, radii=radii,
                         title=f"NBP — RMSE {res.rmse[-1]:.2f} "
                               f"(raw, unaligned frame)"),
            figs / "layout_nbp_raw.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_error_cdf({"NBP": err_nbp, "MDS rigid": err_mds}, bound=peb,
                   ax=axes[0], title="per-node error")
    plot_error_vs_degree({"NBP": err_nbp, "MDS rigid": err_mds}, degrees,
                         ax=axes[1], title="error vs connectivity")
    fig.tight_layout()
    io.save_fig(fig, figs / "compare_nbp_mds_crlb.png")

    # The particle and message diagnostics live in NBP's own frame, so with no
    # anchors it is ground truth that has to move: A is orthogonal, so the
    # alignment inverts exactly. Fitted on the last iteration and reused for
    # all of them, since the frame drifts a little per iteration.
    X_ref = sc.X_true
    if free:
        A, c_hat, c_ref = rigid_transform(res.estimates, sc.targets)
        X_ref = (sc.X_true - c_ref) @ A.T + c_hat

    # Chosen from the data, never hardcoded: the target NBP did worst on.
    worst = t + int(err_nbp.argmax())
    picks = np.unique(np.linspace(0, cfg.n_iter - 1, 6).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    for k, i in enumerate(picks):
        plot_particles(ex["particles_hist"][i], ex["weights_hist"][i],
                       X_ref, sc.num_anchors, [worst], ax=axes.flat[k],
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
                       X_ref, sc.num_anchors, widest.tolist(),
                       bboxes=ex["bboxes"],
                       title="widest anchor-derived priors, iteration 1"),
        figs / "priors.png")

    io.save_fig(
        plot_messages(ex["proposals"], worst, res.particles, res.weights,
                      X_ref, sc.num_anchors, sc.D,
                      title=f"messages into node {worst}, "
                            f"iteration {cfg.n_iter}"),
        figs / "messages_node.png")

    if args.show:
        plt.show()
    plt.close("all")
    print(f"wrote           : {out}")


if __name__ == "__main__":
    main()
