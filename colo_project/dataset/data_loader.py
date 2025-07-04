from pathlib import Path
import numpy as np
from utils.graph_utils import n_hop_distance, nth_hop_adjacency
from utils.metrics import crlb, per_node_peb


class LocalizationScneario:
    def __init__(self, noise, npz_path: Path, n_hop: int = 2, n_adj: int = 1):
        data = np.load(npz_path, allow_pickle=True)
        self.X_true: np.ndarray = data["X_true"]
        self.full_D: np.ndarray = data["full_D"]
        self.D: np.ndarray = data["D"]
        self.B: np.ndarray = data["B"]
        self.RSS: np.ndarray = data["RSS"]
        self.num_anchors = int(data["num_anchors"])
        self.noise = float(data["noise"])

        self.n_hops, self.n_adj = n_hop, n_adj
        self.Dn = n_hop_distance(self.D, self.n_hops)
        self.Bn = nth_hop_adjacency(self.D, self.n_adj)

        if noise is not None:
            self.noise = noise

        self.crlb_cov = crlb(self.B, self.X_true, sigma_db=self.noise)
        self.pebs = per_node_peb(self.crlb_cov)


def generate_test(scneario_name: str):
    from dataset.generate_dataset import generate_scenario
    out_dir = Path(f"./dataset/scenarios/{scneario_name}")
    config_path = Path(f"./dataset/scenarios/{scneario_name}" + ".json")
    generate_scenario(config_path, out_dir)


if __name__ == "__main__":
    generate_test("test")
    data_loc = Path("./dataset/scenarios/test/test_seed31.npz")
    scenario = LocalizationScneario(data_loc)
    Dn = n_hop_distance(scenario.D, 2)
    Bn = nth_hop_adjacency(scenario.D, 1)

    """ crlb_cov = crlb(Bn, scenario.X_true, sigma_db=1)
    pebs = per_node_peb(crlb_cov)
    print(pebs) """