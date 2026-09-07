"""Bounded boxes: the anchor-derived prior region for each node.

A node within range `D[i, j]` of anchor `j` lies inside the axis-aligned box of
half-width `D[i, j]` centred on that anchor. Intersecting those boxes over all
anchors a node hears gives a cheap, conservative prior region to draw its
initial particles from.

Boxes are stored flat as `[min_0, max_0, min_1, max_1, ...]`, i.e. shape
(N, 2*d), matching the `limits` convention used throughout the project.
"""

import numpy as np


def full_area_bbox(n_nodes: int, limits: np.ndarray) -> np.ndarray:
    """The whole deployment area, repeated for every node.

    Used for the no-priors ablation, so that `create_bbox` does not need a
    `priors` flag: the choice of prior region belongs to the caller.
    """
    return np.tile(np.asarray(limits, dtype=float), (n_nodes, 1))


def create_bbox(
    D: np.ndarray,
    anchors: np.ndarray,
    limits: np.ndarray,
    gap: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Intersect per-anchor boxes into one prior box per node.

    Args:
        D: (N, N) distance matrix, 0 where there is no measurement.
        anchors: (A, d) known anchor positions.
        limits: (2*d,) deployment bounds [min_0, max_0, min_1, max_1, ...].
        gap: minimum box width per axis; narrower boxes are widened by `gap`
            on each side.

    Returns:
        bboxes: (N, 2*d) intersected boxes, always within `limits`.
        n_empty: how many (node, axis) intersections came out empty.

    An empty intersection means the ranges to different anchors are mutually
    inconsistent, which noise can cause. Legacy swapped min and max in that
    case, silently producing a valid-looking but meaningless box; here the axis
    falls back to the full deployment range and the occurrence is counted, so
    the caller can see it.
    """
    limits = np.asarray(limits, dtype=float)
    A, d = anchors.shape
    N = D.shape[0]

    lo_lim = limits[0::2]                      # (d,)
    hi_lim = limits[1::2]                      # (d,)

    d_ia = D[:, :A]                            # (N, A) node -> anchor ranges
    # A node has no usable constraint from an anchor it cannot hear, and an
    # anchor imposes none on itself.
    usable = d_ia > 0                          # (N, A)
    usable[:A][np.diag_indices(min(A, N))] = False

    r = d_ia[:, :, None]                       # (N, A, 1)
    lo = anchors[None, :, :] - r               # (N, A, d)
    hi = anchors[None, :, :] + r

    # Unusable pairs must not constrain the intersection.
    lo = np.where(usable[:, :, None], lo, lo_lim)
    hi = np.where(usable[:, :, None], hi, hi_lim)

    box_lo = np.maximum(lo.max(axis=1), lo_lim)     # (N, d)
    box_hi = np.minimum(hi.min(axis=1), hi_lim)

    empty = box_lo > box_hi
    n_empty = int(empty.sum())
    box_lo = np.where(empty, lo_lim, box_lo)
    box_hi = np.where(empty, hi_lim, box_hi)

    if gap > 0:
        narrow = (box_hi - box_lo) < gap
        box_lo = np.where(narrow, np.maximum(box_lo - gap, lo_lim), box_lo)
        box_hi = np.where(narrow, np.minimum(box_hi + gap, hi_lim), box_hi)

    bboxes = np.empty((N, 2 * d), dtype=float)
    bboxes[:, 0::2] = box_lo
    bboxes[:, 1::2] = box_hi
    return bboxes, n_empty


def bbox_contains(bboxes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N,) bool: is `points[i]` inside `bboxes[i]`? A port sanity check."""
    lo, hi = bboxes[:, 0::2], bboxes[:, 1::2]
    return np.all((points >= lo) & (points <= hi), axis=1)


def bbox_area(bboxes: np.ndarray) -> np.ndarray:
    """(N,) product of side lengths."""
    return np.prod(bboxes[:, 1::2] - bboxes[:, 0::2], axis=1)
