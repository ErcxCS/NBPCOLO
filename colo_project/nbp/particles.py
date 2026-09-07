"""Particle initialisation, proposal distributions, and resampling.

Every function here draws randomness and therefore takes `rng` as a required
keyword argument. There is deliberately no `rng=None` default: silently falling
back to a fresh or global generator is exactly the reproducibility bug this
port exists to remove.

Note that scipy's `gaussian_kde.resample()` draws from the *global* legacy
`np.random` singleton unless it is passed `seed=`. Every call below passes it.
"""

import numpy as np
from scipy.stats import gaussian_kde

from colo_project.nbp.potentials import bbox_prior


def init_particles(bboxes, anchors, n_particles, *, rng):
    """
    Uniform particles inside each node's prior box.

    Anchors get `n_particles` copies of their own known position with uniform
    weights. That is what lets the belief mean be taken over *all* nodes:
    the weighted mean of P copies of a point is that point exactly, so anchor
    rows come back as their true positions and no code downstream needs to
    offset indices by `num_anchors`.

    Returns:
        particles: (N, P, d)
        weights:   (N, P), rows sum to 1
    """
    n_nodes = bboxes.shape[0]
    n_anchors, d = anchors.shape

    particles = np.zeros((n_nodes, n_particles, d))
    particles[:n_anchors] = np.repeat(
        anchors[:, None, :], n_particles, axis=1
    )
    weights = np.ones((n_nodes, n_particles))

    for i in range(n_anchors, n_nodes):
        lo, hi = bboxes[i, 0::2], bboxes[i, 1::2]
        particles[i] = rng.uniform(lo, hi, size=(n_particles, d))
        weights[i] = bbox_prior(bboxes[i])(particles[i])

    # A degenerate prior (all zeros) would produce NaN on normalisation.
    bad = weights.sum(axis=1) <= 0
    weights[bad] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return particles, weights


def random_spread(particles_r, d_ru, sigma, *, rng):
    """
    Isotropic range-annulus proposal, used on the first iteration when there is
    no belief about the receiver yet.

    Returns (d_xy, w_xy): offsets to add to r's particles, and the proposal
    density they were drawn from (uniform, hence 1).
    """
    n = particles_r.shape[0]
    r = d_ru + rng.normal(0.0, sigma, size=n)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n)
    d_xy = np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
    return d_xy, np.ones(n)


def relative_spread(particles_u, particles_r, d_ru, sigma, *, rng,
                    w_floor=1e-7):
    """
    Bearing-aware range-annulus proposal.

    Rather than spreading r's particles uniformly around the annulus, fit a KDE
    to the bearings actually implied by the current beliefs of u and r, and draw
    angles from that. `w_xy` is the proposal density, which the caller divides
    out as an importance correction.

    `w_floor` guards the reciprocal: tail angles can otherwise return a density
    near zero and blow up `1 / w_xy`. Legacy had this guard commented out.
    """
    n = particles_u.shape[0]
    delta = particles_u - particles_r
    angles = np.arctan2(delta[:, 1], delta[:, 0])

    # Angles are periodic; replicating at +/-2pi keeps the KDE from treating
    # the wrap-around at +/-pi as a hard boundary.
    extended = np.concatenate([angles, angles + 2 * np.pi, angles - 2 * np.pi])
    kde = gaussian_kde(extended)

    samples = kde.resample(n, seed=rng).T
    samples = np.mod(samples + np.pi, 2 * np.pi) - np.pi
    w_xy = np.maximum(kde(samples.T), w_floor)

    r = (d_ru + rng.normal(0.0, sigma, size=n)).reshape(-1, 1)
    d_xy = np.column_stack([r * np.cos(samples), r * np.sin(samples)])
    return d_xy, w_xy


def resample_indices(weights, n, *, rng):
    """
    Multinomial resampling. Kept (rather than the lower-variance systematic
    scheme) so results stay comparable to the legacy implementation; swapping it
    is a one-line change once the port itself is trusted.
    """
    return rng.choice(weights.size, size=n, replace=True, p=weights)


def split_budget(total: int, k: int) -> np.ndarray:
    """
    Split `total` particles across `k` senders, reproducing the legacy
    floor-divide-and-decrement allocation exactly but as a pure function of the
    count rather than of thread arrival order.

    split_budget(500, 3) -> [166, 167, 167]
    """
    if k <= 0:
        return np.zeros(0, dtype=int)
    alloc = np.empty(k, dtype=int)
    remaining, left = total, k
    for j in range(k):
        alloc[j] = remaining // left
        remaining -= alloc[j]
        left -= 1
    return alloc
