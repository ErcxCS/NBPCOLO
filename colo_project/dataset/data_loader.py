from pathlib import Path
import numpy as np
from utils.graph_utils import n_hop_distance, nth_hop_adjacency
from utils.metrics import crlb, per_node_peb
from mds.classic_mds import ClassicMDS
from utils.metrics import euclidean_metrics
from utils.visualizations import plot_results


class LocalizationScneario:
    def __init__(self,
                 npz_path: Path,
                 noise: float = None,
                 n_hop: int = 2,
                 n_adj: int = 1,
                 run_mds: bool = True):
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
        self.anchors = self.X_true[:self.num_anchors]

        if noise is not None:
            self.noise = noise

        self.crlb_cov = crlb(self.B, self.X_true, sigma_db=self.noise)
        self.pebs = per_node_peb(self.crlb_cov)

        if run_mds:
            mds = ClassicMDS(
                dim=self.X_true.shape[1]
            )
            self.mds_xhat, self.mds_registered = mds.run_mds(
                self.X_true,
                self.D,
                self.full_D,
                self.num_anchors,
                use_full_D=False
            )
            self.mds_rmse, mds_mae, mds_med = euclidean_metrics(
                self.X_true,
                self.mds_registered)
            print(f"MDS RMSE: {self.mds_rmse}")
            plot_results(self.X_true, self.mds_xhat, self.num_anchors, show_anchors=False, show_lines=True)
            plot_results(self.X_true, self.mds_registered, self.num_anchors, show_anchors=False, show_lines=True)




def generate_test(scneario_name: str) -> LocalizationScneario:
    from dataset.generate_dataset import generate_scenario
    scenario_dir = f"./dataset/scenarios/{scneario_name}"
    out_dir = Path(scenario_dir)
    config_path = Path(scenario_dir + ".json")
    scenario_path = generate_scenario(config_path, out_dir)
    scenario = LocalizationScneario(
        npz_path=scenario_path,
        noise=None,
        n_hop=2,
        n_adj=1
    )
    return scenario


if __name__ == "__main__":
    scenario = generate_test("test")