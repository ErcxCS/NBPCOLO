from pathlib import Path
import numpy as np
from utils.graph_utils import n_hop_distance, nth_hop_adjacency
from utils.metrics import crlb, per_node_peb

from utils.graph_utils import get_distance_matrix


class LocalizationScneario:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.X_true: np.ndarray = data["X_true"]
        self.full_D: np.ndarray = data["full_D"]
        self.D: np.ndarray = data["D"]
        self.B: np.ndarray = data["B"]
        self.RSS: np.ndarray = data["RSS"]
        self.num_anchors = int(data["num_anchors"])


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

    crlb_cov = crlb(Bn, scenario.X_true, sigma_db=0.71)
    print(per_node_peb(crlb_cov))
    """ cov_crlb = crlb(scenario.B, scenario.X_true)
    pebs = per_node_peb(cov_crlb)
    print(pebs) """