import argparse
import json
from pathlib import Path
import numpy as np

from colo_project.utils.graph_utils import generate_anchors, generate_targets
from colo_project.utils.graph_utils import get_distance_matrix
from colo_project.constants import ALPHA, D0, STREAM_DATA


def generate_scenario(config_path: Path, out_dir: Path) -> Path:
    """
    Create and save a graph for one scenario defined by a JSON config.

    Saves a .npz file containing:
        - X_true: Ground truth positions (N x d)
        - full_D: Full N x N distance matrix
        - D: Thresholded distance matrix (N x N)
        - B: Binary adjacency matrix (N x N)
        - RSS: Received signal strength within binary adjacency
        - num_anchors: first num_anchors in X_true are anchors
        - noise: shadowing sigma in dB used to generate D
        - alpha, d0: path-loss parameters used to generate D
        - seed: the seed the scenario was generated from

    Returns the path of the saved .npz.
    """
    # Load scenario config
    cfg = json.loads(config_path.read_text())

    seed = cfg.get("seed", None)
    num_nodes = cfg.get("num_nodes", 100)
    d_dim = cfg.get("d_dim", 2)
    num_anchors = cfg.get("num_anchors", 0)
    meters = cfg.get("meters", 100)
    radius = cfg.get("radius", 20)
    noise = cfg.get("noise", 1.0)
    placement = cfg.get("placement", True)
    heterogeneity = cfg.get("heterogeneity", False)
    power_level = cfg.get("power_level", -1)
    symetric = cfg.get("symetric", True)
    # -1: 0 default
    # 0: Bluetooth (0 - 4)
    # 1: Wi-Fi (15 - 20)
    # 2: Cellular (20 - 23)
    # 3: sensor networks (-3 - 14) (Zigbee vs.)
    # 4: RFID (-20 - 0)
    transmission_powers = {
        -1: (0, 0),
        0: (0, 4),
        1: (15, 20),
        2: (20, 23),
        3: (-3, 14),
        4: (-20, 0)
    }
    rng = np.random.default_rng([seed, STREAM_DATA])

    # Generate ground truth positions
    X_true, area = generate_targets(
        num_nodes, d_dim,
        meters,
        rng=rng
    )

    # Anchor generation logic if needed
    if num_anchors > 0:
        if placement:
            anchors = generate_anchors(area, num_anchors, np.sqrt(meters)*1)
            num_anchors = len(anchors)
            X_true[:num_anchors] = anchors
        anchors = X_true[:num_anchors]

    full_D, D, B, RSS = get_distance_matrix(
        X_true, radius, noise=noise,
        alpha=ALPHA, d0=D0,
        heterogeneity=heterogeneity,
        power_level=transmission_powers[power_level],
        symetric=symetric,
        rng=rng
    )

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config_path.stem}_seed{seed}.npz"

    # Save raw measuremnets
    np.savez_compressed(
        out_path,
        X_true=X_true,
        full_D=full_D,
        D=D,
        B=B,
        RSS=RSS,
        num_anchors=num_anchors,
        noise=noise,
        alpha=ALPHA,
        d0=D0,
        seed=seed
    )

    print(f"Saved scenario data to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate localization data scenarios."
    )
    parser.add_argument(
        "--scenarios-dir", type=str, required=True,
        help="Path to JSON scenario config files"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory to save generated .npz files"
    )
    args = parser.parse_args()
    scen_dir = Path(args.scenarios_dir)
    out_dir = Path(args.out_dir)

    for config in sorted(scen_dir.glob("*.json")):
        generate_scenario(config, out_dir)
