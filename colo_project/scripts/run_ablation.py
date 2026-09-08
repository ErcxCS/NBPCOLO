"""Hop-count ablation for NBP.

`n_hop` does two things at once, and this separates them:

  1. Range smoothing. `n_hop_distance` takes a min over paths, so a one-hop
     range can be replaced by a shorter multi-hop detour.
  2. The push set. Nodes reachable within n hops but not heard send negative
     information.

Running with `use_negative=False` keeps (1) and removes (2), so the gap between
the two arms at a given hop count is the contribution of the push term.
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=["dense", "test"])
    parser.add_argument("--radii", nargs="+", type=float, default=[30.0, 20.0],
                        help="comms radius per scenario, same order")
    parser.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "dataset" / "scenarios")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results")
    args = parser.parse_args()

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from colo_project.dataset.data_loader import load_or_generate
    from colo_project.mds.classic_mds import ClassicMDS
    from colo_project.nbp.core import NBP, NBPConfig
    from colo_project.utils import io
    from colo_project.utils.metrics import (
        crlb, euclidean_metrics, summarize_crlb,
    )

    assert len(args.radii) == len(args.scenarios), \
        "--radii must have one entry per scenario"

    out = io.result_dir(args.results_dir, "ablation", 0)
    report = {"git_sha": io.git_sha(), "n_iter": args.n_iter,
              "seeds": args.seeds, "scenarios": {}}

    for name, radius in zip(args.scenarios, args.radii):
        sc = load_or_generate(name, args.scenarios_dir)
        t = sc.num_anchors
        cov = crlb(sc.B, sc.X_true, sc.num_anchors,
                   alpha=sc.alpha, d0=sc.d0, sigma_db=sc.noise)
        crlb_rms = summarize_crlb(cov)["CRLB rms threshold"]
        _, _, rigid = ClassicMDS(dim=sc.dim).run_mds(
            sc.X_true, sc.D, sc.full_D, sc.num_anchors)
        mds_rmse = euclidean_metrics(sc.targets, rigid[t:])[0]

        print(f"\n=== {name}  (CRLB {crlb_rms:.3f}, MDS {mds_rmse:.3f}) ===")
        print(f"{'hop':>4} {'push':>6} {'full RMSE':>18} {'no-push RMSE':>18}")

        rows = {}
        for hop, use_neg in itertools.product(args.hops, [True, False]):
            best, final = [], []
            for seed in args.seeds:
                cfg = NBPConfig(
                    n_iter=args.n_iter, radius=radius, n_hop=hop,
                    use_negative=use_neg, seed=seed,
                )
                r = NBP(sc, cfg).run(verbose=False)
                best.append(float(r.rmse.min()))
                final.append(float(r.rmse[-1]))
            rows[(hop, use_neg)] = {
                "best_mean": float(np.mean(best)),
                "best_std": float(np.std(best)),
                "final_mean": float(np.mean(final)),
                "best_per_seed": best,
            }

        for hop in args.hops:
            f, n = rows[(hop, True)], rows[(hop, False)]
            print(f"{hop:>4} {'':>6} "
                  f"{f['best_mean']:>10.3f} +-{f['best_std']:<5.3f} "
                  f"{n['best_mean']:>10.3f} +-{n['best_std']:<5.3f}")

        report["scenarios"][name] = {
            "radius": radius, "crlb_rms": crlb_rms, "mds_rmse": mds_rmse,
            "results": {f"hop{h}_{'full' if u else 'nopush'}": v
                        for (h, u), v in rows.items()},
        }

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for use_neg, style, lbl in [(True, "o-", "full (with push)"),
                                    (False, "s--", "no push")]:
            m = [rows[(h, use_neg)]["best_mean"] for h in args.hops]
            e = [rows[(h, use_neg)]["best_std"] for h in args.hops]
            ax.errorbar(args.hops, m, yerr=e, fmt=style, capsize=3, label=lbl)
        ax.axhline(mds_rmse, color="tab:orange", ls=":", label="MDS")
        ax.axhline(crlb_rms, color="tab:red", ls="-.", label="CRLB")
        ax.set_xlabel("n_hop")
        ax.set_ylabel("best RMSE over iterations (m)")
        ax.set_xticks(args.hops)
        ax.set_title(f"hop ablation — {name}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        io.save_fig(fig, out / "figures" / f"ablation_{name}.png")
        plt.close(fig)

    io.save_json(out / "ablation.json", report)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
