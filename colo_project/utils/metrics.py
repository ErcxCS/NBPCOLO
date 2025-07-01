import numpy as np


def crlb(B, X_true, alpha=3.15, d0=1.15, sigma=1.0) -> np.ndarray:
    """
    Compute the Cramér–Rao lower bound covariance for node positions.

    Parameters
    ----------
    noisy_RSS : (N,N) array
      Observed RSS in dB (used only to know which pairs exist).
    X_true : (N,2) array
      Ground-truth node positions.
    alpha : float
      Path-loss exponent.
    d0 : float
      Reference distance (in same units as X_true).
    sigma : float
      Standard deviation of additive Gaussian noise in dB.

    Returns
    -------
    cov_crlb : (2N,2N) array
      Minimum covariance matrix for any unbiased estimator of [x1,y1,…,xN,yN].
    """
    rows, cols = np.where(np.triu(B, k=1) == 1)
    pairs = list(zip(rows, cols))
    pairs = [(i, j) for i, j in pairs if np.hypot(*(X_true[i]-X_true[j])) > 0]
    J = jacobian(X_true, pairs, alpha, d0)
    fim = (1/sigma**2) * J.T @ J
    eps = 1e-6 * np.trace(fim) / fim.shape[0]
    cov_crlb = np.linalg.inv(fim + eps*np.eye(fim.shape[0]))
    print(summarize_crlb(cov_crlb))
    return cov_crlb


def jacobian(X_true: np.ndarray, edges, alpha, d0):
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
