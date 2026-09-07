"""Potentials and message algebra for NBP. Pure functions, no RNG.

The interesting one is the positive/negative split. A node that *hears* a
neighbour gets a message concentrated on the range annulus; a node that is
within n-hop reach but does *not* hear a neighbour still learns something --
namely that it is probably far away. That second term is
`negative_information`, and it is what makes multi-hop NBP better than
one-hop NBP.
"""

import numpy as np
from scipy.stats import norm


def bbox_prior(bbox: np.ndarray):
    """
    Uniform pdf over one bounded box.

    Returns a closure `pdf(points) -> density`, zero outside the box.
    """
    lo, hi = np.asarray(bbox)[0::2], np.asarray(bbox)[1::2]
    volume = float(np.prod(hi - lo))

    def pdf(points: np.ndarray) -> np.ndarray:
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        return inside / volume

    return pdf


def range_likelihood(X_r, x_u, d_ru, sigma):
    """
    Gaussian pairwise range potential: how well does each particle of r sit at
    the measured distance `d_ru` from `x_u`?

    Not used by the importance sampler, which draws from the range annulus
    directly and reweights by detection probability. Kept because it is the
    principled likelihood a weighted variant would need.
    """
    dist = np.linalg.norm(X_r - x_u, axis=1)
    return norm.pdf(d_ru - dist, scale=sigma)


def detection_prob(X: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    """
    Probability that a node at each row of `X` (M, d) is heard from `y` (d,).

    Gaussian falloff with the communication radius as its scale. Legacy inlined
    this at three sites in two different spellings.
    """
    diff_sq = np.sum((X - y) ** 2, axis=1)
    return np.exp(-diff_sq / (2.0 * radius ** 2))


def negative_information(
    particles_u: np.ndarray,
    particles_r: np.ndarray,
    weights_r: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Message from a node r that u can reach in >1 hop but does not hear.

    `1 - E_r[P(detect)]` per particle of u: positions of u that *would* have
    been heard by r are down-weighted, because they were not.

    Args:
        particles_u: (M, d) particles being scored.
        particles_r: (P, d) particles of the silent node.
        weights_r: (P,) belief weights of the silent node.
    Returns:
        (M,) message value per particle of u.
    """
    diff_sq = np.sum(
        (particles_u[:, None, :] - particles_r[None, :, :]) ** 2, axis=2
    )
    detect = np.exp(-diff_sq / (2.0 * radius ** 2))
    return 1.0 - detect @ weights_r


def log_message_product(messages, eps: float = 1e-300) -> np.ndarray:
    """
    Product of incoming messages, computed in the log domain.

    A node with 20-40 incoming messages of magnitude ~1e-2 underflows a naive
    np.prod to exactly 0.0 in float64; the subsequent normalisation then yields
    NaN silently. Summing logs and re-centring on the max keeps it finite.

    Returns the product rescaled so its maximum is 1 -- callers normalise
    anyway, and the absolute scale is not meaningful.
    """
    log_m = np.log(np.clip(np.asarray(messages, dtype=float), eps, None))
    total = log_m.sum(axis=0)
    return np.exp(total - total.max())
