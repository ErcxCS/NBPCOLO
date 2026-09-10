import numpy as np
from typing import Tuple, List

from colo_project.constants import ALPHA, D0


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
    num_nodes: int,
    dim: int,
    deployment_area: float,
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    generate node positions within deployment area centered at the origin.

    Args:
        num_nodes: number of nodes to generate.
        dim: spatial dimensions (usually 2).
        deployment_area: half-length of the square area.
        rng: generator for the position draws (required).

    Returns:
        X: np.ndarray shape (num_nodes, dim) of node coordinates.
        bounds: tuple (x_min, x_max, y_min, y_max).
    """
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
    noise: float = 1.0,
    alpha: float = ALPHA,
    d0: float = D0,
    heterogeneity: bool = False,
    power_level: tuple[int, int] = (0, 0),
    symetric: bool = True,
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distance-based and RSS measurement matrices for nodes.

    Args:
        X_true: true positions (N x dim).
        communication_radius: max distance for connectivity (above -> no link).
        noise: zero-mean Gaussian sigma in dB for RSS (0 -> no noise).
        alpha: path-loss exponent.
        d0: reference distance for RSS model.
        rng: generator for the shadowing and power draws (required).

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
        P_i = rng.uniform(power_level[0], power_level[1], N)

    RSS = distance_to_RSS(P_i, full_D, alpha, d0)
    if symetric:
        # Symmetrize and zero diagonal
        RSS = (RSS + RSS.T) / 2.0

    simulated_D, noisy_RSS = RSS_to_distance(
        P_i,
        RSS.copy(),
        alpha,
        d0,
        noise,
        symetric,
        rng=rng
    )

    # Connectivity distance matrix
    D = simulated_D.copy()
    D[D > communication_radius] = 0.0
    B = (D > 0).astype(int)

    return full_D, D, B, noisy_RSS * B


def distance_to_RSS(P_i, full_D, alpha, d0):
    with np.errstate(divide='ignore'):
        RSS = P_i[:, None] - 10.0 * alpha * np.log10(full_D / d0)
    np.fill_diagonal(RSS, 0)
    return RSS


def RSS_to_distance(P_i, RSS, alpha, d0, sigma, symetric, *, rng):
    """
    Invert the path-loss model back to distance, adding log-normal shadowing.

    Shadowing is log-normal in linear power, i.e. zero-mean Gaussian in dB, so
    `sigma` is a dB standard deviation. This is what metrics.crlb assumes.
    """
    if sigma > 0:
        noise_mtx = rng.normal(loc=0.0, scale=sigma, size=RSS.shape)

        if symetric:
            # Mirror the upper triangle instead of averaging with the
            # transpose: averaging would shrink the per-entry sigma by
            # 1/sqrt(2) and silently break the CRLB, which is handed `sigma`.
            iu = np.triu_indices_from(noise_mtx, k=1)
            noise_mtx[(iu[1], iu[0])] = noise_mtx[iu]

        np.fill_diagonal(noise_mtx, 0)
        RSS += noise_mtx

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


def hops_to_complete(D: np.ndarray, max_hops: int = None) -> int:
    """Fewest hops at which every pair of nodes is reachable.

    This is the `n` that makes the n-hop matrix a *complete* distance matrix,
    with no zero (unreachable) entries left off the diagonal -- the hop count
    `ClassicMDS.run_mds` escalates to, since classic MDS needs a full matrix
    and cannot work around holes.

    Runs the same min-plus recurrence as `n_hop_distance` but carries the DP
    forward instead of restarting it per hop, so finding `n` costs one pass
    rather than the `O(n^2)` passes an escalating caller would spend.

    Raises `ValueError` if the graph is disconnected, since then no hop count
    completes the matrix and the honest answer is not a number.
    """
    W = D.astype(float)
    W[W == 0] = np.inf
    np.fill_diagonal(W, 0.0)
    n = len(W)
    max_hops = n - 1 if max_hops is None else max_hops

    reach = W.copy()
    frontier = W.copy()
    for hop in range(1, max_hops + 1):
        if hop > 1:
            frontier = np.min(frontier[:, :, None] + W[None, :, :], axis=1)
            reach = np.minimum(reach, frontier)
        if np.isfinite(reach).all():
            return hop
    unreachable = int((~np.isfinite(reach)).sum())
    raise ValueError(
        f"graph is disconnected: {unreachable} ordered pairs are unreachable "
        f"within {max_hops} hops, so no hop count completes the matrix")


def compare_hop_distances(D: np.ndarray, full_D: np.ndarray, hops) -> list:
    """Per-hop quality of the n-hop matrix against the true distances.

    `n_hop` buys coverage and pays for it in accuracy, and this is the
    measurement of that trade. Each entry reports, for one hop count:

    `coverage`      fraction of off-diagonal pairs that have any value at all;
    `bias_m`        mean signed error against `full_D`, over covered pairs.
                    Two effects fight here and the number says which wins.
                    Geometry pushes multi-hop distances *up*: a path through
                    intermediate nodes is a polyline, never shorter than the
                    straight line it spans. Noise pushes them *down*: the DP
                    takes a minimum over many noisy path sums, and a minimum
                    over noisy candidates preferentially selects the ones
                    whose noise ran negative. On these datasets the selection
                    effect wins and the n-hop matrix comes out systematically
                    *short*, so do not assume the sign;
    `rmse_m`/`mae_m`, `rel_bias`  the same error, squared / absolute /
                    normalised by the true distance;
    `new_bias_m`    bias restricted to pairs this hop *added* over the
                    previous one, which is where the detour error actually
                    lives -- pooled with the direct edges it is diluted;
    `direct_replaced`  fraction of measured one-hop edges whose value the
                    min-plus DP overwrote with a shorter detour. Noise alone
                    makes a detour look shorter often enough to matter, which
                    is why `NBP` keeps the direct range for one-hop pairs
                    rather than taking `D_hop` at face value.

    `hops` is any iterable of hop counts; pass `hops_to_complete(D)` as the
    last one to see the fully-connected matrix. Pure measurement: no RNG, and
    nothing here is fed back into an estimator.
    """
    off = ~np.eye(len(D), dtype=bool)
    direct = (D != 0) & off
    rows = []
    prev_cov = None
    for hop in hops:
        Dn = n_hop_distance(D, hop)
        cov = (Dn != 0) & off
        err = Dn[cov] - full_D[cov]
        row = {
            "hop": int(hop),
            "coverage": float(cov.sum() / off.sum()),
            "n_pairs": int(cov.sum()),
            "bias_m": float(err.mean()),
            "mae_m": float(np.abs(err).mean()),
            "rmse_m": float(np.sqrt((err ** 2).mean())),
            "rel_bias": float((err / full_D[cov]).mean()),
            "direct_replaced": float(
                ((Dn != D) & direct).sum() / direct.sum()),
        }
        new = cov & ~prev_cov if prev_cov is not None else None
        row["new_pairs"] = 0 if new is None else int(new.sum())
        row["new_bias_m"] = (
            None if new is None or not new.any()
            else float((Dn[new] - full_D[new]).mean()))
        rows.append(row)
        prev_cov = cov
    return rows
