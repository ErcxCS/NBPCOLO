"""Nonparametric belief propagation for cooperative localization.

Each node carries a particle cloud approximating its position belief. Per
iteration:

  Phase A (messages)  For every one-hop pair (r -> u), shift r's particles onto
      the measured range annulus, weight them by the probability that u would
      detect them and by the cavity belief of r (its belief with u's own
      previous message divided out), then fit a weighted KDE. Each receiver
      draws a share of a fixed particle budget from each incoming proposal.

  Phase B (beliefs)   Each node multiplies the messages it received. One-hop
      neighbours contribute their KDE evaluated at the candidate particles;
      nodes that are within n-hop reach but were *not* heard contribute
      negative information. The product is normalised and resampled back down
      to `n_particles`.

Anchors are fixed: their particles are copies of their known position, so the
belief mean over *all* nodes returns anchor positions exactly and no index
arithmetic offsets by `num_anchors` anywhere below.

Runs serially and deterministically. Threads were removed: the hot path is
scipy's `gaussian_kde`, whose Cython kernel does not release the GIL, so the
legacy ThreadPoolExecutor bought little while making results depend on thread
arrival order.
"""

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import gaussian_kde

from colo_project.constants import STREAM_NBP
from colo_project.dataset.data_loader import Scenario
from colo_project.nbp.bbox import create_bbox, full_area_bbox
from colo_project.nbp.particles import (
    init_particles, random_spread, relative_spread, resample_indices,
    split_budget,
)
from colo_project.nbp.potentials import (
    detection_prob, log_message_product, negative_information,
)
from colo_project.utils.graph_utils import n_hop_distance
from colo_project.utils.metrics import euclidean_metrics, range_sigma


@dataclass(frozen=True)
class NBPConfig:
    n_particles: int = 125
    n_iter: int = 10
    n_batches: int = 4
    radius: float = 20.0
    n_hop: int = 2
    meters: float = 100.0
    use_priors: bool = True
    # Ablation: drop the negative (push) messages while still using the n-hop
    # matrix for range smoothing, so the two effects of n_hop can be separated.
    use_negative: bool = True
    seed: int = 0


@dataclass
class NBPState:
    particles: np.ndarray   # (N, P, d)
    weights: np.ndarray     # (N, P), rows sum to 1
    incoming: np.ndarray    # (N, N, P): incoming[u, r] = msg r->u at u's parts


@dataclass
class NBPResult:
    scenario: str
    seed: int
    config: dict
    estimates: np.ndarray          # (N_t, d) final target estimates
    particles: np.ndarray          # (N, P, d) final belief clouds
    weights: np.ndarray            # (N, P)
    estimates_hist: np.ndarray     # (n_iter, N_t, d)
    rmse: np.ndarray               # (n_iter,)
    mae: np.ndarray
    med: np.ndarray
    spread: np.ndarray             # (n_iter,) mean per-node belief spread
    n_degenerate: int = 0
    n_empty_bbox: int = 0
    runtime_s: float = 0.0
    # Diagnostics the plots need, recorded rather than recomputed:
    #   particles_hist (n_iter, N, P, d), weights_hist (n_iter, N, P),
    #   spread_hist (n_iter, N_t), bboxes (N, 2d), proposals {(r,u): kde}.
    # ~3 MB at N=100, P=125, n_iter=10, linear in every knob; at N=1000,
    # P=500 it is ~400 MB and wants a config gate.
    extras: dict = field(default_factory=dict)


class NBP:
    def __init__(self, scenario: Scenario, cfg: NBPConfig):
        self.sc = scenario
        self.cfg = cfg

        # Three matrices with three distinct jobs, kept separate on purpose:
        #
        #   D_direct  measured one-hop ranges. The evidence a positive message
        #             is built from.
        #   D_hop     n-hop reachability. A nonzero entry means the pair takes
        #             part in the negative (push) term; for pairs that are only
        #             reachable via a detour its value is a shortest-path sum,
        #             never a measurement.
        #   C         one-hop adjacency; picks who pulls versus who pushes.
        #
        # A one-hop pair keeps its own measurement in D_hop: the min-plus DP
        # would otherwise replace it with a shortest-path detour whenever noise
        # makes the detour look shorter (~31% of one-hop pairs on `dense`), so
        # restore those entries from D_direct. Only the values change; every
        # one-hop pair is nonzero either way, so the reachability gate is
        # untouched.
        self.D_direct = scenario.D
        self.D_hop = n_hop_distance(scenario.D, cfg.n_hop)
        self.C = scenario.B
        direct = self.C == 1
        self.D_hop[direct] = self.D_direct[direct]
        self.n_anchors = scenario.num_anchors
        self.N = scenario.n_nodes
        self.d = scenario.dim

        half = cfg.meters / 2.0
        self.limits = np.array([-half, half] * self.d, dtype=float)
        self.n_empty_bbox = 0

    # -- setup ------------------------------------------------------------

    def _init_state(self, rng) -> NBPState:
        # Kept on self only so the priors can be plotted afterwards; the value
        # handed to init_particles is unchanged.
        if self.cfg.use_priors:
            self.bboxes, self.n_empty_bbox = create_bbox(
                self.sc.D, self.sc.anchors, self.limits
            )
        else:
            self.bboxes = full_area_bbox(self.N, self.limits)

        particles, weights = init_particles(
            self.bboxes, self.sc.anchors, self.cfg.n_particles, rng=rng
        )
        incoming = np.ones((self.N, self.N, self.cfg.n_particles))
        return NBPState(particles, weights, incoming)

    def _sigma(self, d_ru: float) -> float:
        """Ranging sigma at the measured distance, from the same noise model
        the CRLB uses -- so the bound is the true floor for this sampler."""
        return float(
            max(range_sigma(d_ru, self.sc.noise, self.sc.alpha), 1e-6)
        )

    # -- one iteration ----------------------------------------------------

    def _approximate_messages(self, state, mu, it, rng):
        """Phase A. Returns the KDE proposals and each node's candidate pool.

        Looping receiver-outer means every write lands in a slot that depends
        only on the graph, never on iteration order -- which is what makes the
        particle budget reproducible.
        """
        cfg = self.cfg
        budget = cfg.n_batches * cfg.n_particles
        proposals = {}
        sampled = [None] * self.N

        for u in range(self.n_anchors, self.N):
            senders = np.flatnonzero(self.C[u])
            senders = senders[senders != u]
            if senders.size == 0:
                continue

            alloc = split_budget(budget, senders.size)
            offsets = np.concatenate([[0], np.cumsum(alloc)])
            pool = np.empty((budget, self.d))

            for j, r in enumerate(senders):
                # A sender is one-hop by construction, so this is that pair's
                # own measurement: __init__ restores the direct reading over
                # whatever the min-plus DP produced.
                #
                # Letting the detour win was measured and is better on RMSE
                # (dense best 2.488 vs 2.808) because the min is a one-sided
                # noise filter -- any detour sum is >= the true distance -- but
                # it substitutes a shortest path for a measurement and adds
                # downward bias (-0.65m vs -0.18m on dense), which the CRLB, an
                # unbiased-estimator bound, does not account for.
                d_ru = self.D_hop[r, u]
                sigma = self._sigma(d_ru)
                particles_r = state.particles[r]
                if it == 0:
                    d_xy, w_xy = random_spread(
                        particles_r, d_ru, sigma, rng=rng
                    )
                else:
                    d_xy, w_xy = relative_spread(
                        state.particles[u], particles_r, d_ru, sigma, rng=rng
                    )

                X_ru = particles_r + d_xy
                detect = detection_prob(X_ru, mu[u], cfg.radius)

                # Cavity: r's belief with the message u sent it divided out,
                # then corrected for the proposal it was drawn from.
                cavity = state.weights[r] / state.incoming[r, u]
                W_ru = detect * cavity / w_xy
                total = W_ru.sum()
                if not np.isfinite(total) or total <= 0:
                    W_ru = np.full(cfg.n_particles, 1.0 / cfg.n_particles)
                else:
                    W_ru = W_ru / total

                kde = gaussian_kde(
                    X_ru.T, weights=W_ru, bw_method='silverman'
                )
                proposals[(r, u)] = kde
                pool[offsets[j]:offsets[j + 1]] = kde.resample(
                    alloc[j], seed=rng
                ).T

            sampled[u] = pool

        return proposals, sampled

    def _update_beliefs(self, state, proposals, sampled, rng):
        """Phase B. Returns a new state; the input is never mutated."""
        cfg = self.cfg
        particles = state.particles.copy()
        weights = state.weights.copy()
        incoming = state.incoming.copy()
        n_degenerate = 0

        for u in range(self.n_anchors, self.N):
            pool = sampled[u]
            if pool is None:
                continue

            all_msgs, one_hop, senders = [], [], []
            for r in np.flatnonzero(self.D_hop[u]):
                if r == u:
                    continue
                if self.C[u, r] == 1 and (r, u) in proposals:
                    msg = proposals[(r, u)](pool.T)
                    one_hop.append(msg)
                else:
                    if not cfg.use_negative:
                        continue
                    # Reachable within n hops but not heard: absence of a
                    # detection is itself evidence about where u is not.
                    msg = negative_information(
                        pool, state.particles[r], state.weights[r], cfg.radius
                    )
                all_msgs.append(msg)
                senders.append(r)

            if not all_msgs:
                continue

            W_u = log_message_product(all_msgs)
            if one_hop:
                denom = np.sum(one_hop, axis=0)
                W_u = np.divide(
                    W_u, denom,
                    out=np.zeros_like(W_u), where=denom > 0,
                )

            total = W_u.sum()
            if not np.isfinite(total) or total <= 0:
                n_degenerate += 1
                W_u = np.full(pool.shape[0], 1.0 / pool.shape[0])
            else:
                W_u = W_u / total

            idx = resample_indices(W_u, cfg.n_particles, rng=rng)
            particles[u] = pool[idx]
            w_new = W_u[idx]
            weights[u] = w_new / w_new.sum()
            for r, msg in zip(senders, all_msgs):
                incoming[u, r] = msg[idx]

        return NBPState(particles, weights, incoming), n_degenerate

    # -- driver -----------------------------------------------------------

    def run(self, verbose: bool = True) -> NBPResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        state = self._init_state(np.random.default_rng([cfg.seed, STREAM_NBP]))

        hist, rmse, mae, med, spread = [], [], [], [], []
        # Diagnostics for the plots. Recording only -- nothing below draws
        # from `rng`, and nothing is inserted between two RNG consumers, so
        # the trace stays bitwise identical at a fixed seed.
        particles_hist, weights_hist, spread_hist = [], [], []
        proposals = {}
        n_degenerate = 0
        t = self.n_anchors

        for it in range(cfg.n_iter):
            rng = np.random.default_rng([cfg.seed, STREAM_NBP, it])
            mu = np.einsum('ijk,ij->ik', state.particles, state.weights)

            proposals, sampled = self._approximate_messages(
                state, mu, it, rng
            )
            state, deg = self._update_beliefs(state, proposals, sampled, rng)
            n_degenerate += deg

            mu = np.einsum('ijk,ij->ik', state.particles, state.weights)
            est = mu[t:]
            r_, a_, m_ = euclidean_metrics(self.sc.targets, est)
            hist.append(est.copy())
            rmse.append(r_)
            mae.append(a_)
            med.append(m_)
            sp = _belief_spread(state.particles[t:], state.weights[t:], est)
            spread_hist.append(sp)
            spread.append(sp.mean())
            particles_hist.append(state.particles.copy())
            weights_hist.append(state.weights.copy())

            if verbose:
                print(f"  iter {it + 1:>2}/{cfg.n_iter}  "
                      f"RMSE {r_:7.3f}  med {m_:7.3f}  spread {spread[-1]:7.3f}")

        return NBPResult(
            scenario=self.sc.name,
            seed=self.sc.seed,
            config=asdict(cfg),
            estimates=hist[-1],
            particles=state.particles,
            weights=state.weights,
            estimates_hist=np.array(hist),
            rmse=np.array(rmse),
            mae=np.array(mae),
            med=np.array(med),
            spread=np.array(spread),
            n_degenerate=n_degenerate,
            n_empty_bbox=self.n_empty_bbox,
            runtime_s=time.perf_counter() - t0,
            extras={
                "particles_hist": np.array(particles_hist),
                "weights_hist": np.array(weights_hist),
                "spread_hist": np.array(spread_hist),
                "bboxes": self.bboxes,
                # Last iteration only, and RAM only: a gaussian_kde does not
                # go into an .npz, so the message figure is its record.
                "proposals": proposals,
            },
        )


def _belief_spread(particles_t, weights_t, estimates_t) -> np.ndarray:
    """Per-node sqrt(trace) of the weighted particle covariance.

    Takes already-sliced target arrays and asserts they line up, so the
    anchor/target misalignment that legacy's weighted_covariance had cannot be
    expressed here.
    """
    assert particles_t.shape[0] == estimates_t.shape[0], (
        "particles and estimates must cover the same nodes"
    )
    delta = particles_t - estimates_t[:, None, :]
    var = np.einsum('ij,ijk->ik', weights_t, delta ** 2)
    return np.sqrt(var.sum(axis=1))
