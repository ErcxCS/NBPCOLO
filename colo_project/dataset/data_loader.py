from pathlib import Path
import numpy as np

import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt


class LocalizationScneario:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.X_true: np.ndarray = data["X_true"]
        self.full_D: np.ndarray = data["full_D"]
        self.D: np.ndarray = data["D"]
        self.B: np.ndarray = data["B"]
        self.RSS: np.ndarray = data["RSS"]
        self.num_anchors = int(data["num_anchors"])

    def get_graph(self, hop: int = 1) -> np.ndarray:
        A = self.B.copy()
        result = self.D.copy()
        for _ in range(hop - 1):
            A = (A @ self.B) > 0
            result = result | A
        return result.astype(int)


def generate_test(scneario_name: str):
    from dataset.generate_dataset import generate_scenario
    out_dir = Path(f"./dataset/scenarios/{scneario_name}")
    config_path = Path(f"./dataset/scenarios/{scneario_name}" + ".json")
    generate_scenario(config_path, out_dir)


def n_hop_distance(D: np.ndarray, n_hops: int) -> np.ndarray:
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")
    N = D.shape[0]
    # Replace zeros with inf except diagonal
    W = D.astype(float)
    mask = (W == 0)
    W[mask] = np.inf
    np.fill_diagonal(W, 0.0)

    # Dynamic programming: W^1 = W, W^k = min(W^{k - 1} + W)
    D_prev = W.copy()
    Dn = W.copy()
    for _ in range(2, n_hops + 1):
        # min-plus matrix multiplication
        # compute all pairs: min_k (D_prev[i, k] + W[k, j])
        M = D_prev[:, :, None] + W[None, :, :]
        D_prev = np.min(M, axis=1)
        Dn = np.minimum(Dn, D_prev)
    # unreachable => 0
    Dn[Dn == np.inf] = 0.0
    return Dn


def plot_MRF(X: np.ndarray, B: np.ndarray, n_anchors: int, network: np.ndarray, radius: int,):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    G = nx.from_numpy_array(network*B)

    # Plot edges between node of interest and its immediate neighbors
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='gray')
    

    th = 0
    # Highlight intersection areas
    """ for i in range(n_anchors, len(X)):
        if intersections is not None and i < len(intersections):
            if B[i, :n_anchors].sum() > th:
                #if network[i, :n_anchors].sum() > 0:
                int_bbox = intersections[i]
                xmin, xmax, ymin, ymax = int_bbox
                ax.fill_between([xmin, xmax], ymin, ymax, alpha=0.3) """
        

    # Plot anchors
    ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=300)
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')


    plotted = False
    for i in range(n_anchors, len(X)):
        if B[i, :n_anchors].sum() > th:
            if not plotted:
                ax.scatter(X[i, 0], X[i, 1], marker="+", c="g", label=r"$N_{t}$", s=75)
                plotted = True
            else:
                 ax.scatter(X[i, 0], X[i, 1], marker="+", c="g", s=75)
            ax.annotate(rf"$T_{{{i - n_anchors}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='g')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title("Network Coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_test("test")
    loc = LocalizationScneario(Path("./dataset/scenarios/test/test_seed31.npz"))
    network, B, C = get_n_hop(loc.D, 7, 1)
    other_network = n_hop_distance(loc.D, 7)
    print(np.allclose(network, other_network))
    
    print(network)
    print(np.count_nonzero(network))
    print(len(loc.X_true))
    plot_MRF(loc.X_true, B, loc.num_anchors, network, 20)