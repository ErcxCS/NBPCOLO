"""Compare raw MDS, cold NBP and warm-start NBP by Procrustes disparity.

Every layout here is scored and drawn *unregistered* -- exactly as the
estimator produced it, with no Procrustes onto the truth and no anchor
registration. That is what makes the three comparable at all: with no
anchors none of them shares a frame with the truth or with each other, so a
metre-valued RMSE would be measuring three different arbitrary frames rather
than three estimates. `metrics.procrustes_disparity` removes translation,
rotation, reflection and scale internally, and what is left is shape
agreement alone.

Read it beside a metre score, never instead of one: disparity is
dimensionless and scale-blind, so a layout that is uniformly 20% too large
still scores 0. `run_nbp` is where the metre-valued numbers live.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="barcelona_noanchor")
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "dataset" / "scenarios")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "results")
    parser.add_argument("--n-particles", type=int, default=125)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--n-hop", type=int, default=2)
    parser.add_argument("--meters", type=float, default=100.0)
    parser.add_argument("--warm-halfwidth", type=float, default=None,
                        help="seed box half-width in m (default: radius/2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    from colo_project.dataset.data_loader import load_or_generate
    from colo_project.mds.classic_mds import ClassicMDS
    from colo_project.nbp.core import NBP, NBPConfig
    from colo_project.utils import io
    from colo_project.utils.metrics import (
        procrustes_disparity, procrustes_hist,
    )
    from colo_project.utils.visualizations import (
        plot_convergence, plot_raw_layouts,
    )

    sc = load_or_generate(args.scenario, args.scenarios_dir)
    t = sc.num_anchors

    mds = ClassicMDS(dim=sc.dim)
    x_hat, _, rigid = mds.run_mds(sc.X_true, sc.D, sc.full_D, sc.num_anchors)
    # The seed for the warm start, and the MDS layout being scored, are the
    # same object on purpose: `run_mds` returns `rigid=None` when there are no
    # anchors to register against, and the raw embedding is then the only
    # layout MDS has. Never the truth-aligned one -- that is the frame the
    # estimator is supposed to recover.
    mds_seed = x_hat if rigid is None else rigid

    def build(warm: bool):
        cfg = NBPConfig(
            n_particles=args.n_particles, n_iter=args.n_iter,
            n_batches=args.n_batches, radius=args.radius, n_hop=args.n_hop,
            meters=args.meters, seed=args.seed,
            warm_start=warm, warm_halfwidth=args.warm_halfwidth,
        )
        return NBP(sc, cfg, init_positions=mds_seed)

    print(f"scenario        : {sc.name} (seed {sc.seed})")
    print(f"nodes / anchors : {sc.n_nodes} / {sc.num_anchors}")
    print(f"mean degree     : {sc.mean_degree:.2f}")
    print("NBP cold:")
    cold = build(False).run()
    print("NBP warm:")
    warm_nbp = build(True)
    warm = warm_nbp.run()

    # Targets only, per the scoring convention -- anchors are known, and with
    # anchors present they would also drag the shared Procrustes scaling
    # towards the part of the layout that was never estimated.
    layouts = {
        "MDS (raw)": mds_seed[t:],
        "NBP cold (raw)": cold.estimates,
        "NBP warm start (raw)": warm.estimates,
    }
    disparity = {k: procrustes_disparity(v, sc.targets)
                 for k, v in layouts.items()}
    hist = {"NBP cold": procrustes_hist(cold.estimates_hist, sc.targets),
            "NBP warm start": procrustes_hist(warm.estimates_hist,
                                              sc.targets)}

    print("\nProcrustes disparity vs ground truth (raw, unaligned; "
          "0 = same shape):")
    for label, value in disparity.items():
        print(f"  {label:<22} {value:.5f}")
    for label, curve in hist.items():
        print(f"  {label:<22} best {curve.min():.5f} "
              f"at iter {curve.argmin() + 1}")

    out = io.result_dir(args.results_dir, sc.name, sc.seed)
    io.save_json(out / "metrics_procrustes.json", {
        "scenario": sc.name,
        "seed": sc.seed,
        "algorithm": "procrustes-comparison",
        "git_sha": io.git_sha(),
        "config": cold.config,
        "warm_halfwidth_used": warm_nbp.warm_halfwidth,
        "note": ("Procrustes disparity M^2 on raw unregistered layouts, "
                 "targets only. Removes translation/rotation/reflection and "
                 "scale, so it is dimensionless and scale-blind: read it "
                 "beside the metre-valued RMSE in metrics_nbp.json."),
        "disparity": disparity,
        "disparity_hist": {k: v.tolist() for k, v in hist.items()},
        "rmse_raw": {"NBP cold": cold.rmse.tolist(),
                     "NBP warm start": warm.rmse.tolist()},
    })
    io.save_arrays(
        out / "arrays_procrustes.npz",
        mds_raw=mds_seed, cold_estimates=cold.estimates,
        warm_estimates=warm.estimates,
        cold_hist=hist["NBP cold"], warm_hist=hist["NBP warm start"],
    )

    figs = out / "figures"
    io.save_fig(
        plot_raw_layouts(
            layouts, sc.targets, disparities=disparity,
            title=f"{sc.name} seed {sc.seed} — raw layouts, no alignment"),
        figs / "procrustes_layouts.png")
    io.save_fig(
        plot_convergence(
            hist, baselines={"MDS (raw)": disparity["MDS (raw)"]},
            ylabel="Procrustes $M^2$  (lower = better)", logy=True,
            title=f"{sc.name} — shape agreement with truth"),
        figs / "procrustes_tracking.png")

    print(f"wrote           : {out}")

    if args.show:
        from matplotlib import pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
