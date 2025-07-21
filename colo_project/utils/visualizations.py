from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx


def plot_MRF(X: np.ndarray,
             B: np.ndarray,
             n_anchors: int,
             network: np.ndarray,
             radius: int,):
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
    ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1],
               marker="*",
               c="r",
               label=r"$N_{a}$",
               s=300)
    
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X[i, 0], X[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=12,
                    color='r'
                    )

    plotted = False
    for i in range(n_anchors, len(X)):
        if B[i, :n_anchors].sum() > th:
            if not plotted:
                ax.scatter(X[i, 0], X[i, 1],
                           marker="+", c="g", label=r"$N_{t}$", s=75)
                plotted = True
            else:
                ax.scatter(X[i, 0], X[i, 1], marker="+", c="g", s=75)
            ax.annotate(rf"$T_{{{i - n_anchors}}}$", (X[i, 0], X[i, 1]),
                        textcoords="offset points",
                        xytext=(0, 10), ha='center',
                        fontsize=12, color='g')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title("Network Coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results(X, X_hat, num_anchors: int,
                 show_lines=False,
                 show_anchors=False):
    plt.scatter(X[:, 0], X[:, 1], label="True X")
    plt.scatter(X_hat[:, 0], X_hat[:, 1], label="Predicted Points")
    plt.legend()
    if show_anchors:
        plt.scatter(X[:num_anchors, 0], X[:num_anchors, 1], "ro")
        plt.scatter(X_hat[:num_anchors, 0], X_hat[:num_anchors, 1], "go")
    if show_lines:
        for i in range(len(X)):
            plt.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "y--")
    plt.show()
