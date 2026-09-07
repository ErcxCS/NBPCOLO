from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """
    One generated localization scenario, loaded from a .npz.

    Pure data: loading a Scenario computes nothing and plots nothing.
    Algorithms take a Scenario and return their own result object; derived
    quantities (n-hop graphs, CRLB) are computed by whoever needs them, because
    NBP and MDS disagree about which hop count they want.

    Node ordering: the first `num_anchors` rows of every (N, ...) array are
    anchors with known positions; the rest are targets to localize.
    """

    name: str
    seed: int
    X_true: np.ndarray      # (N, d) ground truth positions
    full_D: np.ndarray      # (N, N) true pairwise distances
    D: np.ndarray           # (N, N) noisy distances, 0 beyond comms radius
    B: np.ndarray           # (N, N) binary adjacency
    RSS: np.ndarray         # (N, N) received signal strength, masked by B
    num_anchors: int
    noise: float            # shadowing sigma in dB
    alpha: float            # path-loss exponent used to generate D
    d0: float               # path-loss reference distance used to generate D

    @property
    def anchors(self) -> np.ndarray:
        return self.X_true[:self.num_anchors]

    @property
    def targets(self) -> np.ndarray:
        return self.X_true[self.num_anchors:]

    @property
    def n_nodes(self) -> int:
        return self.X_true.shape[0]

    @property
    def n_targets(self) -> int:
        return self.n_nodes - self.num_anchors

    @property
    def dim(self) -> int:
        return self.X_true.shape[1]

    @property
    def mean_degree(self) -> float:
        return float(self.B.sum(axis=1).mean())

    @classmethod
    def load(cls, npz_path: Path, name: str = None) -> "Scenario":
        data = np.load(npz_path, allow_pickle=True)
        seed = int(data["seed"])
        if name is None:
            # Strip the "_seed<n>" suffix the generator appends.
            name = Path(npz_path).stem.removesuffix(f"_seed{seed}")
        return cls(
            name=name,
            seed=seed,
            X_true=data["X_true"],
            full_D=data["full_D"],
            D=data["D"],
            B=data["B"],
            RSS=data["RSS"],
            num_anchors=int(data["num_anchors"]),
            noise=float(data["noise"]),
            alpha=float(data["alpha"]),
            d0=float(data["d0"]),
        )


def load_or_generate(name: str, scenarios_dir: Path) -> Scenario:
    """
    Load scenario `name`, generating its .npz from `<name>.json` if missing.

    Shared by the run_* scripts so neither has to know the layout convention:
    config at `<scenarios_dir>/<name>.json`, output under
    `<scenarios_dir>/<name>/<name>_seed<seed>.npz`.
    """
    from colo_project.dataset.generate_dataset import generate_scenario

    scenarios_dir = Path(scenarios_dir)
    config_path = scenarios_dir / f"{name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No scenario config at {config_path}")

    out_dir = scenarios_dir / name
    existing = sorted(out_dir.glob(f"{name}_seed*.npz"))
    if not existing:
        return Scenario.load(generate_scenario(config_path, out_dir), name=name)
    return Scenario.load(existing[0], name=name)
