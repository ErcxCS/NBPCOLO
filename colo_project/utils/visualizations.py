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
    ax.set_title("Network Coverage")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_results(X, X_hat, num_anchors: int,
                 show_lines=False,
                 show_anchors=False,
                 ax=None,
                 title=None):
    """
    True vs. estimated positions, optionally joined by error lines.

    Returns the figure; the caller decides whether to savefig or show.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(X[:, 0], X[:, 1], label="True X")
    ax.scatter(X_hat[:, 0], X_hat[:, 1], label="Predicted Points")
    if show_anchors:
        ax.scatter(X[:num_anchors, 0], X[:num_anchors, 1],
                   c="r", marker="*", s=200, label="Anchors (true)")
        ax.scatter(X_hat[:num_anchors, 0], X_hat[:num_anchors, 1],
                   c="g", marker="*", s=200, label="Anchors (est.)")
    if show_lines:
        for i in range(len(X)):
            ax.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "y--")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    return fig
