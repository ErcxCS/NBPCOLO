import numpy as np
from typing import Optional, Tuple, List


def generate_anchors(
    deployment_area: Tuple[float, float, float, float],
    anchor_count: int,
    border_offset: float = 0.0
) -> np.ndarray:
    """
    Generate anchor node positions in a hexagonal grid within the given area.

    Args:
        deployment_area: (x_min, x_max, y_min, y_max) bounds of the field.
        anchor_count: total number of anchor points to generate.
        border_offset: distance to inset anchors from the deployment boundary.

    Returns:
        anchors: np.ndarray of shape (anchor_count, 2) with (x, y) positions.
    """
    x_min, x_max, y_min, y_max = deployment_area
    x_min += border_offset
    x_max -= border_offset
    y_min += border_offset
    y_max -= border_offset

    # Determine grid resolution
    points_per_axis = int(np.ceil(np.sqrt(anchor_count)))
    if points_per_axis**2 < anchor_count:
        points_per_axis += 1

    x_coords = np.linspace(x_min, x_max, points_per_axis)
    y_coords = np.linspace(y_min, y_max, points_per_axis)

    anchors: List[Tuple[float, float]] = []
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            # shift every other row for hex pattern
            x1, x2 = x_coords[1], x_coords[0]
            x_shifted = x + (0.5 * (x1 - x2) if j % 2 == 0 else 0)
            if len(anchors) < anchor_count:
                if x_min <= x_shifted <= x_max and y_min <= y <= y_max:
                    anchors.append((x_shifted, y))
    return np.array(anchors)


def generate_targets(
    seed: Optional[int],
    num_nodes: int,
    dim: int,
    deployment_area: float
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    generate node positions within deployment area centered at the origin.

    Args:
        seed: random seed (None for RNG default).
        num_nodes: number of nodes to generate.
        dim: spatial dimensions (usually 2).
        deployment_area: half-length of the square area.

    Returns:
        X: np.ndarray shape (num_nodes, dim) of node coordinates.
        bounds: tuple (x_min, x_max, y_min, y_max).
    """
    rng = np.random.default_rng(seed)
    half = deployment_area / 2.0
    bounds = (-half, half, -half, half)
    x_min, x_max, y_min, y_max = bounds

    X = rng.uniform(
        low=(x_min, y_min),
        high=(x_max, y_max),
        size=(num_nodes, dim)
    )
    return X, bounds


def get_distance_matrix(
    X_true: np.ndarray,
    communication_radius: float,
    noise: float = 0.0,
    alpha: float = 3.15,
    d0: float = 1.15,
    heterogeneity: bool = False,
    power_level: tuple[int, int] = (0, 0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distance-based and RSS measurement matrices for nodes.

    Args:
        X_true: true positions (N x dim).
        communication_radius: max distance for connectivity (above -> no link).
        noise: log-normal noise sigma for RSS (0 -> no noise).
        alpha: path-loss exponent.
        d0: reference distance for RSS model.

    Returns:
        full_D: full Euclidean distance matrix (N x N).
        D: thresholded distances (0 where > communication_radius).
        B: binary adjacency (1 where D>0, else 0).
        RSS: received signal strength matrix (in dB), with optional noise.
    """
    # Full pairwise distances
    diffs = X_true[:, None, :] - X_true[None, :, :]
    full_D = np.linalg.norm(diffs, axis=-1)

    # Path-loss model baseline power per node
    N = X_true.shape[0]
    if not heterogeneity:
        P_i = np.zeros(N)
        P_i.fill(power_level[0])
    else:
        P_i = np.random.uniform(power_level[0], power_level[1], N)

    RSS = distance_to_RSS(P_i, full_D, alpha, d0)
    simulated_D, noisy_RSS = RSS_to_distance(P_i, RSS, alpha, d0, noise)

    # Connectivity distance matrix
    D = simulated_D.copy()
    D[D > communication_radius] = 0.0
    B = (D > 0).astype(int)

    return full_D, D, B, RSS


def distance_to_RSS(P_i, full_D, alpha, d0):
    with np.errstate(divide='ignore'):
        RSS = P_i[:, None] - 10.0 * alpha * np.log10(full_D / d0)
    np.fill_diagonal(RSS, 0)
    return RSS


def RSS_to_distance(P_i, RSS, alpha, d0, sigma):
    if sigma > 0:
        noise_matrix = np.random.lognormal(mean=0, sigma=sigma, size=RSS.shape)
        RSS += noise_matrix

    with np.errstate(divide='ignore', invalid='ignore'):
        D = d0 * 10 ** ((P_i[:, None] - RSS) / (10 * alpha))
    np.fill_diagonal(D, 0)
    return D, RSS


def n_hop_distance(D: np.ndarray, n_hops: int) -> np.ndarray:
    """
    Compute the shortest-path distance matrix allowing up to n_hops edges.

    Args:
        D: direct-distance matrix (N x N), zero where no direct edge.
        n_hops: maximum number of hops (>=1)

    Returns:
        Dn: distance matrix (N x N) where
            Dn[i, j] = shortest sum of edge distances among all paths
            from i to j using most n_hops edges; 0 if unreachable.
    """
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")
    W = D.astype(float)
    mask = (W == 0)
    W[mask] = np.inf
    np.fill_diagonal(W, 0.0)

    # DP: W^1 = W, W^k = min(W^{k^1} + W)
    D_prev = W.copy()
    Dn = W.copy()
    for _ in range(2, n_hops + 1):
        # compute all pairs: min_k (D_prev[i, k] + W[k, j])
        M = D_prev[:, :, None] + W[None, :, :]
        D_prev = np.min(M, axis=1)
        Dn = np.minimum(Dn, D_prev)

    Dn[Dn == np.inf] = 0.0
    return Dn


def nth_hop_adjacency(D: np.ndarray, n_hops: int) -> np.ndarray:
    """
    Compute binay adjacency matrix for nth hop

    Args:
        D: direct-distance matrix (N x N), zero where no direct edge.
        n_hops: hop number for adjacency (>=1)

    Returns:
        Bn: binary matrix where Bn[i, j] = 1 if shortest path uses n_hops edge.
    """
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")
    Dn = n_hop_distance(D, n_hops)
    Bn = Dn > 0
    return Bn.astype(int)
