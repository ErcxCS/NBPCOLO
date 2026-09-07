"""Run the classic-MDS baseline on a scenario and persist metrics + figures."""

import argparse
from pathlib import Path

import matplotlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="test",
                        help="scenario name, i.e. <scenarios-dir>/<name>.json")
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "dataset" / "scenarios")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--show", action="store_true",
                        help="also open the figures interactively")
    args = parser.parse_args()

    # Must precede the first pyplot import, which happens inside
    # colo_project.utils.visualizations.
    if not args.show:
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    from colo_project.dataset.data_loader import load_or_generate
    from colo_project.mds.classic_mds import ClassicMDS
    from colo_project.utils import io
    from colo_project.utils.metrics import (
        crlb, euclidean_metrics, per_node_peb, summarize_crlb,
    )
    from colo_project.utils.visualizations import plot_results

    sc = load_or_generate(args.scenario, args.scenarios_dir)

    cov = crlb(sc.B, sc.X_true, sc.num_anchors,
               alpha=sc.alpha, d0=sc.d0, sigma_db=sc.noise)
    crlb_rms = summarize_crlb(cov)["CRLB rms threshold"]
    peb = per_node_peb(cov)

    mds = ClassicMDS(dim=sc.dim)
    x_hat, affine, rigid = mds.run_mds(
        sc.X_true, sc.D, sc.full_D, sc.num_anchors
    )

    # Score on targets only: anchor positions are known, so including them
    # dilutes the error with exact zeros.
    t = sc.num_anchors
    rmse_affine, mae_affine, med_affine = euclidean_metrics(
        sc.targets, affine[t:])
    rmse_rigid, mae_rigid, med_rigid = euclidean_metrics(
        sc.targets, rigid[t:])

    print(f"scenario        : {sc.name} (seed {sc.seed})")
    print(f"nodes / anchors : {sc.n_nodes} / {sc.num_anchors}")
    print(f"mean degree     : {sc.mean_degree:.2f}")
    print(f"noise (dB)      : {sc.noise}")
    print(f"CRLB rms        : {crlb_rms:.3f}")
    print(f"MDS RMSE affine : {rmse_affine:.3f}")
    print(f"MDS RMSE rigid  : {rmse_rigid:.3f}")

    out = io.result_dir(args.results_dir, sc.name, sc.seed)
    io.save_json(out / "metrics.json", {
        "scenario": sc.name,
        "seed": sc.seed,
        "algorithm": "classic_mds",
        "git_sha": io.git_sha(),
        "n_nodes": sc.n_nodes,
        "num_anchors": sc.num_anchors,
        "mean_degree": sc.mean_degree,
        "noise_db": sc.noise,
        "alpha": sc.alpha,
        "d0": sc.d0,
        "crlb_rms": crlb_rms,
        "rmse_affine": rmse_affine,
        "mae_affine": mae_affine,
        "med_affine": med_affine,
        "rmse_rigid": rmse_rigid,
        "mae_rigid": mae_rigid,
        "med_rigid": med_rigid,
    })
    io.save_arrays(out / "arrays_mds.npz",
                   x_hat=x_hat, affine=affine, rigid=rigid, peb=peb)

    fig = plot_results(sc.X_true, rigid, sc.num_anchors,
                       show_lines=True, show_anchors=True,
                       title=f"MDS (rigid) — RMSE {rmse_rigid:.2f}")
    fig.savefig(out / "layout_mds.png", dpi=120)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"wrote           : {out}")


if __name__ == "__main__":
    main()
