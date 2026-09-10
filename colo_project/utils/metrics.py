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
            "singular (translation/rotation) and no absolute bound exists. "
            "Use crlb_anchor_free for the gauge-projected bound instead."
        )

    # FIM over target coordinates only, then invert
    fim = _fim(B, X_true, alpha, d0, sigma_db)
    targets = np.arange(2 * num_anchors, 2 * X_true.shape[0])
    fim = fim[np.ix_(targets, targets)]

    return np.linalg.inv(fim)


def _fim(B, X_true, alpha=ALPHA, d0=D0, sigma_db=1):
    """Fisher information over every node coordinate, `(2N, 2N)`.

    Shared by both bounds: they differ only in how they dispose of the
    translation/rotation gauge that leaves this matrix rank-deficient by 3.
    """
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

    return J.T @ Rinv @ J


def gauge_basis(X_true: np.ndarray) -> np.ndarray:
    """The FIM's exact nullspace, `(2*N, 3)` in 2D, ordered x0,y0,x1,y1,...

    A range is unchanged by translating or rotating the whole network, so the
    FIM annihilates precisely these three directions: the two translations and
    the rotation generator evaluated at `X_true`. Scale is *not* among them --
    ranges are metric -- which is why an anchor-free bound still exists.

    The rotation generator is taken about the centroid so that it comes out
    orthogonal to the translations; the span is the same either way.
    """
    N, d = X_true.shape
    if d != 2:
        raise ValueError(f"gauge basis is written for 2D, got d={d}")
    G = np.zeros((2 * N, 3))
    G[0::2, 0] = 1.0                                   # translate x
    G[1::2, 1] = 1.0                                   # translate y
    G[0::2, 2] = -(X_true[:, 1] - X_true[:, 1].mean())  # rotate
    G[1::2, 2] = X_true[:, 0] - X_true[:, 0].mean()
    return G


def crlb_anchor_free(B, X_true, alpha=ALPHA, d0=D0, sigma_db=1):
    """Gauge-projected CRLB covariance over *all* coordinates, `(2N, 2N)`.

    With no anchors there is no absolute frame, so the FIM is singular by
    construction and `crlb` above refuses. The answer is not to regularize it
    -- a Tikhonov eps would make the "bound" a function of eps rather than of
    geometry -- but to drop the three directions the measurements cannot see.
    `Q` spans the orthogonal complement of `gauge_basis`, `Q.T @ fim @ Q` is
    full rank and inverts exactly, and mapping back gives the minimum-variance
    bound on the network *shape*. There is no rank tolerance anywhere: the
    deficiency is known analytically to be 3, not discovered from a threshold.

    Compare it against a Procrustes-aligned error, never a raw one -- it says
    nothing about the absolute frame, because nothing can.
    """
    fim = _fim(B, X_true, alpha, d0, sigma_db)
    G = gauge_basis(X_true)
    Q = np.linalg.qr(G, mode="complete")[0][:, G.shape[1]:]
    return Q @ np.linalg.inv(Q.T @ fim @ Q) @ Q.T


def rigid_transform(X_hat: np.ndarray, X_ref: np.ndarray):
    """Best rotation/reflection + shift taking `X_hat` onto `X_ref`.

    Returned as `(A, c_hat, c_ref)` and applied as `(X - c_hat) @ A + c_ref`,
    so the map fitted on the position estimates can be reused on the particle
    clouds behind them -- and, since `A` is orthogonal, inverted as
    `(X - c_ref) @ A.T + c_hat` to bring ground truth into the estimator's own
    frame instead.

    No scaling: ranges fix the scale, so allowing it would flatter the
    estimate for free. Reflection *is* allowed, because a range-only network
    and its mirror image are genuinely indistinguishable.
    """
    from scipy.linalg import orthogonal_procrustes
    c_hat, c_ref = X_hat.mean(axis=0), X_ref.mean(axis=0)
    A, _ = orthogonal_procrustes(X_hat - c_hat, X_ref - c_ref)
    return A, c_hat, c_ref


def align_rigid(X_hat: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """`X_hat` mapped onto `X_ref` by `rigid_transform`.

    The anchor-free counterpart of `ClassicMDS.register_anchors_rigid`, which
    solves the same problem against the anchor rows alone. Use it to score an
    anchor-free run, where raw error is mostly gauge.
    """
    A, c_hat, c_ref = rigid_transform(X_hat, X_ref)
    return (X_hat - c_hat) @ A + c_ref


def procrustes_disparity(X_hat: np.ndarray, X_ref: np.ndarray) -> float:
    """Procrustes disparity `M^2` between an estimate and a reference.

    `scipy.spatial.procrustes` centres both configurations, scales each to
    unit Frobenius norm and rotates one onto the other, then returns the
    residual sum of squares: `0` when the two shapes match exactly, `1` when
    they have nothing in common. A mirror image scores `0`, as it does under
    `rigid_transform` -- a range-only network and its reflection are
    genuinely indistinguishable.

    This removes *scale* on top of the rigid gauge, so it measures shape and
    nothing else, and it is dimensionless: it cannot say whether a node is
    2 m or 40 m out, and a layout that is uniformly 20% too large scores a
    perfect 0 even though ranges fix the scale and that is a real error. It
    is the right tool for comparing raw, unregistered layouts, which
    `euclidean_metrics` cannot do at all; it is the wrong tool for anything
    that has to be read against the CRLB. Report it beside a metre-valued
    score, never instead of one.

    Argument order matches `align_rigid`: estimate first, reference second.
    The reference is what gets handed to scipy first, so every estimate is
    scored against an identically normalised reference -- the disparity is
    symmetric only to floating point, so the order is fixed here rather than
    left to the caller.
    """
    from scipy.spatial import procrustes
    return float(procrustes(X_ref, X_hat)[2])


def procrustes_hist(estimates_hist, X_ref: np.ndarray) -> np.ndarray:
    """`procrustes_disparity` per iteration, as an `(n_iter,)` array.

    The per-iteration counterpart used to track shape convergence, which is
    what the legacy prototype plotted as "Procrustes Similarity". Note that
    lower is better, unlike the name.
    """
    return np.array([procrustes_disparity(e, X_ref) for e in estimates_hist])


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
