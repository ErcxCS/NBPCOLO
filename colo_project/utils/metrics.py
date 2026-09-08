import numpy as np

from colo_project.constants import ALPHA, D0


def per_node_error(targets: np.ndarray, predicts: np.ndarray) -> np.ndarray:
    """Per-node Euclidean error, `(N_t,)`.

    The aggregates below hide which nodes are bad; the plots need the vector.
    """
    return np.sqrt(np.sum((targets - predicts)**2, axis=1))


def euclidean_metrics(targets: np.ndarray, predicts: np.ndarray):
    """
    Compute a handful of Euclidean‐error metrics:
      - RMSE
      - MAE  (mean absolute error)
      - MedAE (median absolute error)
    """
    errors = per_node_error(targets, predicts)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    med = np.median(errors)
    # print(f"RMSE: {rmse}, MAE: {mae}, MED: {med}")
    return rmse, mae, med


def range_sigma(d, sigma_db, alpha=ALPHA):
    """
    Convert a dB shadowing sigma into a ranging sigma at distance d.

    Inverting the log-distance path-loss model maps a dB perturbation onto a
    multiplicative distance error, so the ranging sigma grows linearly with d:
        sigma_d = (ln10 / (10*alpha)) * d * sigma_db

    Shared by the CRLB and the NBP proposal so both use one noise model.
    """
    return (np.log(10) / (10 * alpha)) * np.asarray(d) * sigma_db


def crlb(B, X_true, num_anchors, alpha=ALPHA, d0=D0, sigma_db=1):
    """
    Anchored CRLB covariance for the target coordinates.

    Range-only measurements are invariant to translation and rotation, so the
    full 2N x 2N FIM is always rank-deficient by 3 (in 2D) and cannot be
    inverted. Anchors resolve that gauge freedom: their positions are known, so
    they are dropped from the parameter vector rather than estimated. What is
    left is full rank and inverts exactly, with no regularization needed.

    Returns cov of shape (2*N_targets, 2*N_targets), ordered x0,y0,x1,y1,...
    over the target nodes only (i.e. X_true[num_anchors:]).
    """
    if num_anchors <= 0:
        raise ValueError(
            "CRLB needs at least one anchor: without anchors the FIM is "
            "singular (translation/rotation) and no absolute bound exists."
        )

    # 1) pick your links
    rows, cols = np.where(np.triu(B, 1) == 1)
    pairs = list(zip(rows, cols))

    # 2) build Jacobian and per-link distance sigmas
    dij = np.array(
        [np.linalg.norm(X_true[i] - X_true[j]) for (i, j) in pairs]
    )
    J = jacobian(X_true, pairs, alpha=alpha, d0=d0)

    # 3) per-link distance noise
    dsigs = range_sigma(dij, sigma_db, alpha)
    Rinv = np.diag(1.0 / dsigs**2)

    # 4) FIM over target coordinates only, then invert
    fim = J.T @ Rinv @ J
    targets = np.arange(2 * num_anchors, 2 * X_true.shape[0])
    fim = fim[np.ix_(targets, targets)]

    return np.linalg.inv(fim)


def jacobian(X_true: np.ndarray, edges, alpha=ALPHA, d0=D0):
    """
    X_true: (N, d) array of ground-truth positions
    edges: list of (i, j) node-pairs for which RSS exists
    alpha, d0: path-loss paramters
    Returns: J of shape (len(edges), d*N)
    """
    N, d = X_true.shape
    M = len(edges)
    ln10 = np.log(10)
    J = np.zeros((M, d*N), dtype=float)

    for k, (i, j) in enumerate(edges):
        xi, yi = X_true[i]
        xj, yj = X_true[j]
        dx = xi - xj
        dy = yi - yj
        dij = np.hypot(dx, dy)
        if dij == 0:
            continue  # avoid division by zero

        # common scalar factor: ∂h/∂d * 1/d
        fac = -10 * alpha / (ln10 * dij**2)
        # fac = 1.0 / dij

        J[k, 2*i] = fac * dx  # ∂h/∂x_i
        J[k, 2*i + 1] = fac * dy  # ∂h/∂y_i
        J[k, 2*j] = -J[k, 2*i]  # ∂h/∂x_j = -∂h/∂x_i
        J[k, 2*j + 1] = -J[k, 2*i + 1]  # ∂h/∂y_j = -∂h/∂y_i

    return J


def summarize_crlb(cov_crlb: np.ndarray):
    """
    Given the full 2N x 2N CRLB covariance, return:
      - rms: RMS per-coordinate bound
      - trace: sum of variances
      - det: determinant of cov_crlb
    """
    trace = np.trace(cov_crlb)
    N2 = cov_crlb.shape[0]
    rms = np.sqrt(trace / N2) * np.sqrt(2)
    # det = np.linalg.det(cov_crlb)
    # return {"trace": trace, "rms": rms, "det": det}
    return {"CRLB rms threshold": rms}


def per_node_peb(cov_crlb: np.ndarray):
    """
    Given the full 2N x 2N CRLB covariance, return
    an array of length N with the position-error bound per node.
    """
    N2 = cov_crlb.shape[0]
    N = N2 // 2
    pebs = np.zeros(N)
    for i in range(N):
        var_x = cov_crlb[2*i,   2*i]
        var_y = cov_crlb[2*i+1, 2*i+1]
        pebs[i] = np.sqrt(var_x + var_y)
    return pebs
