"""Figures for the localization runs.

Every helper returns a Figure and never calls `plt.show()` -- the caller saves
or shows. Every helper also accepts `ax=None`, so several can be composed into
one grid without any of them stealing the figure. The legacy prototype broke
both rules (functions that took an `ax` and then called `plt.show()` anyway),
which is why none of its figures could be combined.

Node indices are absolute indices into the full `(N, ...)` arrays, per the
anchors-first convention -- never `node - num_anchors`. Nodes of interest are
chosen by the caller *from the data* (worst error, widest prior); the legacy
versions hardcoded node 4 or 5 and a zoom window to match.
"""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from colo_project.nbp.potentials import detection_prob

ANCHOR_KW = dict(marker="*", c="tab:red", s=180, zorder=5)
TRUE_KW = dict(marker="P", c="tab:green", s=45)


def _axes(ax, **kw):
    """Return `(fig, ax)`, creating the figure only when none was given."""
    if ax is None:
        fig, ax = plt.subplots(**kw)
    else:
        fig = ax.figure
    return fig, ax


def _finish(ax, title, legend=True):
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize="small")
    ax.grid(alpha=0.3)


def plot_network(X, B, num_anchors, *, ax=None, title=None):
    """Connectivity graph: one-hop edges, anchors starred, targets plus.

    No per-node annotation -- at N=100 the labels are unreadable, which is
    exactly why every legacy variant hardcoded a zoom window instead.
    """
    fig, ax = _axes(ax, figsize=(6, 6))
    G = nx.from_numpy_array(np.asarray(B))
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.8", width=0.6)
    ax.scatter(X[num_anchors:, 0], X[num_anchors:, 1],
               label="targets", **TRUE_KW)
    ax.scatter(X[:num_anchors, 0], X[:num_anchors, 1],
               label="anchors", **ANCHOR_KW)
    ax.set_aspect("equal", adjustable="datalim")
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_results(X, X_hat, num_anchors: int,
                 show_lines=False,
                 show_anchors=False,
                 ax=None,
                 title=None,
                 radii=None):
    """
    True vs. estimated positions, optionally joined by error lines.

    `radii`, when given, is a belief spread per row of `X_hat`, drawn as a
    dashed circle around each estimate -- the estimator's own uncertainty next
    to its actual error. Anchors have zero spread, so pass zeros for them
    rather than making this function offset by `num_anchors`.

    Returns the figure; the caller decides whether to savefig or show.
    """
    fig, ax = _axes(ax, figsize=(6, 6))

    ax.scatter(X[:, 0], X[:, 1], label="True X")
    ax.scatter(X_hat[:, 0], X_hat[:, 1], label="Predicted Points")
    if show_anchors:
        ax.scatter(X[:num_anchors, 0], X[:num_anchors, 1],
                   c="r", marker="*", s=200, label="Anchors (true)")
        ax.scatter(X_hat[:num_anchors, 0], X_hat[:num_anchors, 1],
                   c="g", marker="*", s=200, label="Anchors (est.)")
    if show_lines:
        for i in range(len(X)):
            ax.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "y--")
    if radii is not None:
        for centre, rad in zip(X_hat, radii):
            ax.add_patch(plt.Circle(centre, rad, fill=False,
                                    ls=":", ec="tab:blue", alpha=0.5))
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_raw_layouts(layouts, X_true, *, disparities=None, title=None):
    """Several *unregistered* layouts against the truth, side by side.

    `layouts` maps a label to an `(n, d)` estimate in its own frame -- no
    Procrustes, no anchor registration, nothing moved. Each panel therefore
    shows the gauge as well as the error, which is the whole point: an
    anchor-free estimate can have the right shape while sitting in a frame
    that shares nothing with the truth's, and a panel that had been aligned
    first would hide exactly that.

    No error lines and no anchor markers, unlike `plot_results` at its default
    call: joining a node to its estimate across two unrelated frames draws the
    frame offset, not the error, and at that point every line is the same line.

    `disparities` maps the same labels to `metrics.procrustes_disparity` and
    goes into each panel title -- the number that survives the frame
    difference the panels are there to make visible.
    """
    labels = list(layouts)
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 5.2))
    for ax, label in zip(np.atleast_1d(axes), labels):
        sub = label
        if disparities is not None:
            sub = (f"{label}\n"
                   f"Procrustes $M^2$ = {disparities[label]:.4f}")
        plot_results(X_true, layouts[label], 0, ax=ax, title=sub)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_convergence(curves, *, baselines=None, ax=None, title=None,
                     xlabel="iteration", ylabel="error (m)", logy=False):
    """Per-iteration curves, plus flat reference lines.

    `curves` maps a label to an `(n_iter,)` array; `baselines` maps a label to
    a scalar drawn as an `axhline`. Replaces the separate legacy plotters for
    RMSE, Procrustes similarity and uncertainty, which were the same figure
    three times over.

    `logy` is for curves that span orders of magnitude, which Procrustes
    disparity does -- a failed run sits near 1 and a good one near 1e-3, and
    on a linear axis the failure flattens everything worth comparing onto
    zero. Off by default, so the metre-valued figures are unchanged.
    """
    fig, ax = _axes(ax, figsize=(7, 4.5))
    styles = ["o-", "s--", "^:", "d-."]
    for (label, values), style in zip(curves.items(), styles * 4):
        values = np.asarray(values)
        ax.plot(range(1, len(values) + 1), values, style, label=label,
                alpha=0.85)
    for (label, value), colour in zip((baselines or {}).items(),
                                      ["tab:orange", "tab:red", "tab:purple"]):
        ax.axhline(value, color=colour, ls="-.", label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_error_cdf(errors, *, bound=None, ax=None, title=None):
    """Empirical CDF of per-node error, one curve per label.

    A single RMSE hides whether an estimator wins on the median and loses on
    the tail; this is the figure that shows it. `bound` is the per-node PEB
    from `metrics.per_node_peb`, drawn as its own curve -- the CRLB is a
    per-node bound, so it has a distribution too, not just an rms.
    """
    fig, ax = _axes(ax, figsize=(6, 4.5))
    series = dict(errors)
    if bound is not None:
        series["CRLB (per-node PEB)"] = bound
    for label, values in series.items():
        values = np.sort(np.asarray(values))
        frac = np.arange(1, values.size + 1) / values.size
        ls = "--" if label.startswith("CRLB") else "-"
        ax.step(values, frac, where="post", ls=ls,
                label=f"{label} (med {np.median(values):.2f})")
    ax.set_xlabel("error (m)")
    ax.set_ylabel("fraction of targets")
    ax.set_ylim(0, 1)
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_error_vs_degree(errors, degrees, *, ax=None, title=None):
    """Per-node error against neighbour count, with a quadratic trend.

    Connectivity dominates every algorithmic effect measured so far, so this
    is the figure that says whether a bad node is bad because of the algorithm
    or because it hears almost nobody.
    """
    fig, ax = _axes(ax, figsize=(6, 4.5))
    degrees = np.asarray(degrees, dtype=float)
    grid = np.linspace(degrees.min(), degrees.max(), 100)
    for label, values in errors.items():
        values = np.asarray(values)
        r = np.corrcoef(degrees, values)[0, 1]
        pts = ax.scatter(degrees, values, s=18, alpha=0.6,
                         label=f"{label} (r={r:+.2f})")
        fit = np.poly1d(np.polyfit(degrees, values, 2))
        ax.plot(grid, fit(grid), color=pts.get_facecolor()[0], lw=2)
    ax.set_xlabel("one-hop neighbours")
    ax.set_ylabel("error (m)")
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_particles(particles, weights, X_true, num_anchors, nodes, *,
                   bboxes=None, ax=None, title=None):
    """Belief clouds for selected nodes, sized by weight.

    `nodes` are absolute indices. `bboxes` draws each node's prior rectangle,
    stored flat as `[min_0, max_0, min_1, max_1]` -- the same order the legacy
    code unpacked as `xmin, xmax, ymin, ymax`.
    """
    fig, ax = _axes(ax, figsize=(6, 6))
    ax.scatter(X_true[num_anchors:, 0], X_true[num_anchors:, 1],
               c="0.8", s=12, zorder=1)
    ax.scatter(X_true[:num_anchors, 0], X_true[:num_anchors, 1],
               label="anchors", **ANCHOR_KW)
    for k, node in enumerate(nodes):
        colour = plt.cm.tab10(k % 10)
        w = np.asarray(weights[node])
        pts = np.asarray(particles[node])
        ax.scatter(pts[:, 0], pts[:, 1], s=4 + 400 * w, alpha=0.35,
                   color=colour, label=f"particles of {node}")
        mean = w @ pts
        # Label once: the markers mean the same thing for every node, and a
        # legend entry per node would swamp the axes.
        ax.scatter(*mean, marker="X", s=90, color=colour, ec="k", zorder=6,
                   label="weighted mean" if k == 0 else None)
        ax.scatter(*X_true[node], marker="P", s=70, color=colour, ec="k",
                   zorder=6, label="true position" if k == 0 else None)
        ax.plot([X_true[node, 0], mean[0]], [X_true[node, 1], mean[1]],
                ls="--", color=colour, lw=1)
        if bboxes is not None:
            lo, hi = bboxes[node][0::2], bboxes[node][1::2]
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=colour,
                                   ls=":", lw=1.2))
    # "box", not "datalim": this one gets tiled into a shared-axes grid to
    # show a belief over iterations, and datalim is illegal when axes share.
    ax.set_aspect("equal", adjustable="box")
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_messages(proposals, node, particles, weights, X_true, num_anchors, D,
                  *, max_senders=6, ax=None, title=None):
    """Incoming messages at one receiver, over its own belief.

    `proposals` is the Phase-A dict `{(sender, receiver): gaussian_kde}`. The
    support of each message is scattered directly from the KDE's dataset, so
    nothing is evaluated on a grid. Capped at the `max_senders` nearest
    senders by measured range, because a node on `dense` hears ~21 of them and
    all 21 overlaid is mud.
    """
    fig, ax = _axes(ax, figsize=(7, 6))
    senders = [r for (r, u) in proposals if u == node]
    senders.sort(key=lambda r: D[r, node])
    senders = senders[:max_senders]

    for k, r in enumerate(senders):
        colour = plt.cm.tab10(k % 10)
        kde = proposals[(r, node)]
        support = kde.dataset.T
        ax.scatter(support[:, 0], support[:, 1], s=10, alpha=0.3,
                   color=colour, label=f"msg {r}->{node}  d={D[r, node]:.1f}")
        ax.scatter(*X_true[r], marker="o", s=60, color=colour, ec="k",
                   zorder=6)

    pts = np.asarray(particles[node])
    w = np.asarray(weights[node])
    ax.scatter(pts[:, 0], pts[:, 1], s=4 + 400 * w, c="k", alpha=0.5,
               label=f"belief of {node}")
    ax.scatter(*X_true[node], marker="P", s=140, c="tab:green", ec="k",
               zorder=7, label=f"true {node}")
    ax.set_aspect("equal", adjustable="box")
    _finish(ax, title)
    fig.tight_layout()
    return fig


def plot_detection_model(X, D, node, radius, limits, *, ax=None, title=None):
    """P(detect) around one node, against who it actually heard.

    Uses `potentials.detection_prob`, the same function the sampler weights
    with, so the contour is the model in force rather than a redrawing of it.
    Green nodes were heard, red were not: the mismatch between the two and the
    contour is the shadowing.
    """
    fig, ax = _axes(ax, figsize=(7, 6))
    lo, hi = np.asarray(limits)[0::2], np.asarray(limits)[1::2]
    gx = np.linspace(lo[0], hi[0], 200)
    gy = np.linspace(lo[1], hi[1], 200)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    Z = detection_prob(grid, X[node], radius).reshape(XX.shape)

    cf = ax.contourf(XX, YY, Z, levels=20, cmap="viridis", alpha=0.85)
    fig.colorbar(cf, ax=ax, label="P(detect)")
    heard = np.flatnonzero(D[node] > 0)
    silent = np.setdiff1d(np.arange(len(X)), np.append(heard, node))
    ax.scatter(X[silent, 0], X[silent, 1], marker="x", c="tab:red", s=25,
               label="not heard")
    ax.scatter(X[heard, 0], X[heard, 1], marker="P", c="tab:green", s=35,
               ec="k", lw=0.4, label="heard")
    ax.scatter(*X[node], marker="*", c="w", s=250, ec="k",
               label=f"node {node}")
    ax.set_aspect("equal")
    _finish(ax, title)
    fig.tight_layout()
    return fig


def scenario_figures(sc):
    """The figures that depend only on the scenario, as `{stem: fig}`.

    Returns figures rather than writing them, like every other helper here;
    the caller saves them beside the run directories, since they are identical
    for every run of a scenario.

    `Scenario` is pure data and stores neither the comms radius nor the field
    extent, so both are recovered from the arrays: generation zeroes every
    entry of `D` above the radius, making `D.max()` the threshold itself to
    within one measurement.
    """
    radius = float(sc.D.max())
    lo = sc.X_true.min(axis=0)
    hi = sc.X_true.max(axis=0)
    pad = 0.05 * (hi - lo)
    limits = np.empty(2 * sc.dim)
    limits[0::2], limits[1::2] = lo - pad, hi + pad
    # Representative rather than chosen: the median-degree target.
    degrees = sc.B.sum(axis=1)[sc.num_anchors:]
    node = sc.num_anchors + int(np.argsort(degrees)[degrees.size // 2])

    return {
        "network": plot_network(
            sc.X_true, sc.B, sc.num_anchors,
            title=f"{sc.name} — {sc.n_nodes} nodes, "
                  f"mean degree {sc.mean_degree:.1f}"),
        "model_detection": plot_detection_model(
            sc.X_true, sc.D, node, radius, limits,
            title=f"detection model at node {node} "
                  f"(degree {int(sc.B[node].sum())})"),
        "model_rss": plot_rss_model(
            sc.alpha, sc.d0, sc.noise, radius,
            title=f"{sc.name} — path loss and shadowing"),
    }


def plot_rss_model(alpha, d0, sigma_db, radius, *, ax=None, title=None):
    """The path-loss curve the scenario was generated from.

    RSS = -10*alpha*log10(d/d0) at 0 dBm, with the shadowing band that
    `generate_dataset` injects. The `+/-1 sigma` band maps onto a *ranging*
    error that grows linearly with distance, which is why the CRLB rises with
    range and why the comms radius is marked.
    """
    fig, ax = _axes(ax, figsize=(6.5, 4.5))
    d = np.linspace(d0, max(radius * 2.5, d0 * 2), 400)
    rss = -10.0 * alpha * np.log10(d / d0)
    ax.plot(d, rss, color="tab:blue", lw=2, label=f"alpha={alpha}, d0={d0}")
    for k, a in ((1, 0.28), (2, 0.14)):
        ax.fill_between(d, rss - k * sigma_db, rss + k * sigma_db,
                        color="tab:blue", alpha=a,
                        label=f"+/-{k} sigma ({k * sigma_db:g} dB)")
    ax.axvline(radius, color="tab:red", ls="--",
               label=f"comms radius {radius:g} m")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("RSS (dB, 0 dBm TX)")
    _finish(ax, title)
    fig.tight_layout()
    return fig
