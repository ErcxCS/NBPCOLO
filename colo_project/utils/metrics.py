import numpy as np


def crlb(B, X_true, alpha=3.15, d0=1.15, sigma_db=1):
    # 1) pick your links
    rows, cols = np.where(np.triu(B, 1) == 1)
    pairs = list(zip(rows, cols))

    # 2) build Jacobian and per-link distance sigmas
    dij_list = []
    for (i, j) in pairs:
        d = np.linalg.norm(X_true[i] - X_true[j])
        dij_list.append(d)
    J = jacobian(X_true, pairs)  # uses fac=1/d

    # 3) per-link distance noise
    dsigs = np.array([(np.log(10)/(10*alpha))*d * sigma_db for d in dij_list])
    Rinv = np.diag(1.0 / dsigs**2)

    # 4) FIM and inversion
    fim = J.T @ Rinv @ J
    eps = 1e-6 * np.trace(fim)/fim.shape[0]
    cov_crlb = np.linalg.inv(fim + eps*np.eye(fim.shape[0]))

    summary = summarize_crlb(cov_crlb)  # now uses "rms2d"
    print(summary)
    return cov_crlb


def jacobian(X_true: np.ndarray, edges, alpha=3.15, d0=1.15):
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
    det = np.linalg.det(cov_crlb)
    return {"trace": trace, "rms": rms, "det": det}


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
