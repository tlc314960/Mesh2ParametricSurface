"""Matplotlib-based debug visualization (saved to PNG, no display needed)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .swept_tube_fitter import FitResult, ring_polylines  # noqa: E402


def _set_equal_aspect(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_debug_figure(
    out_path: str,
    points: np.ndarray,
    result: FitResult,
    max_points: int = 4000,
):
    """Render a multi-panel debug image.

    Panels:
      * sampled points + endpoints + initial vs final centerline
      * fitted cross-section rings colored by reliability
      * proxy surface wireframe vs input points
      * reliability profile (2D)
    """
    ep_a, ep_b = result.endpoints
    cl_final = result.centerline.samples
    cl_init = result.initial_centerline

    # Subsample input points for plotting speed.
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    fig = plt.figure(figsize=(18, 12))

    # Panel 1: points + endpoints + centerlines.
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="lightgray", alpha=0.4)
    ax1.plot(cl_init[:, 0], cl_init[:, 1], cl_init[:, 2],
             "b--", lw=1.5, label="initial centerline")
    ax1.plot(cl_final[:, 0], cl_final[:, 1], cl_final[:, 2],
             "r-", lw=2.5, label="final centerline")
    ax1.scatter(*ep_a.point, c="green", s=80, marker="o", label="endpoint A")
    ax1.scatter(*ep_b.point, c="magenta", s=80, marker="o", label="endpoint B")
    ax1.set_title("Points + endpoints + centerlines")
    ax1.legend(loc="upper right", fontsize=8)
    _set_equal_aspect(ax1, pts)

    # Panel 2: cross-section rings colored by reliability.
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="lightgray", alpha=0.2)
    rings = ring_polylines(result.sections, n_theta=result.config.angular_resolution)
    cmap = plt.get_cmap("RdYlGn")
    for ring, sec in zip(rings, result.sections):
        loop = np.vstack([ring, ring[:1]])
        ax2.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                 color=cmap(sec.reliability), lw=1.3)
    ax2.plot(cl_final[:, 0], cl_final[:, 1], cl_final[:, 2], "k-", lw=1.0)
    ax2.set_title("Cross-section rings (green=reliable, red=unreliable)")
    _set_equal_aspect(ax2, pts)

    # Panel 3: proxy surface vs input points.
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="steelblue", alpha=0.3)
    surf = result.surface
    if len(surf.faces) > 0:
        sv = surf.vertices
        ax3.plot_trisurf(sv[:, 0], sv[:, 1], sv[:, 2],
                         triangles=surf.faces, color="orange",
                         alpha=0.4, edgecolor="none")
    ax3.set_title("Proxy surface vs input points")
    _set_equal_aspect(ax3, pts)

    # Panel 4: reliability + radius profile.
    ax4 = fig.add_subplot(2, 2, 4)
    if result.sections:
        us = [s.u for s in result.sections]
        rel = [s.reliability for s in result.sections]
        rad = [s.radius for s in result.sections]
        cov = [s.angular_coverage for s in result.sections]
        ax4.plot(us, rel, "g-o", ms=3, label="reliability")
        ax4.plot(us, cov, "b-^", ms=3, label="angular coverage")
        ax4.axhline(result.config.min_reliability, color="r", ls="--",
                    lw=1, label="min reliability")
        ax4b = ax4.twinx()
        ax4b.plot(us, rad, "m-s", ms=3, alpha=0.6, label="radius")
        ax4b.set_ylabel("radius", color="m")
        ax4.set_xlabel("u (along centerline)")
        ax4.set_ylabel("score")
        ax4.set_ylim(0, 1.05)
        ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("Per-section reliability / coverage / radius")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_revolve_debug_figure(
    out_path: str,
    points: np.ndarray,
    result,
    max_points: int = 4000,
):
    """Render a multi-panel debug image for a surface-of-revolution fit.

    Panels:
      * sampled points + estimated axis
      * input points vs revolved proxy surface
      * 2D profile r(h) with per-slice radii colored by reliability
      * per-slice reliability / angular coverage
    """
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    o = result.axis_origin
    a = result.axis_dir
    seg = result.axis_segment

    fig = plt.figure(figsize=(18, 12))

    # Panel 1: points + axis.
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="lightgray", alpha=0.4)
    if len(seg) == 2:
        ax1.plot(seg[:, 0], seg[:, 1], seg[:, 2], "r-", lw=3, label="rotation axis")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Points + estimated revolution axis")
    _set_equal_aspect(ax1, pts)

    # Panel 2: proxy surface vs input.
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="steelblue", alpha=0.3)
    surf = result.surface
    if len(surf.faces) > 0:
        sv = surf.vertices
        ax2.plot_trisurf(sv[:, 0], sv[:, 1], sv[:, 2],
                         triangles=surf.faces, color="orange",
                         alpha=0.4, edgecolor="none")
    if len(seg) == 2:
        ax2.plot(seg[:, 0], seg[:, 1], seg[:, 2], "r-", lw=2)
    ax2.set_title("Revolved proxy surface vs input points")
    _set_equal_aspect(ax2, pts)

    # Panel 3: 2D profile r(h).
    ax3 = fig.add_subplot(2, 2, 3)
    rel = points - o
    h_all = rel @ a
    rad_all = np.linalg.norm(rel - np.outer(h_all, a), axis=1)
    # Subsample scatter for speed.
    if len(h_all) > max_points:
        sidx = np.random.default_rng(1).choice(len(h_all), max_points, replace=False)
    else:
        sidx = np.arange(len(h_all))
    ax3.scatter(h_all[sidx], rad_all[sidx], s=2, c="lightgray", alpha=0.4,
                label="input points")
    ax3.plot(result.profile_h, result.profile_r, "b-", lw=2.5, label="fitted profile r(h)")
    if result.samples:
        sh = [s.h for s in result.samples]
        sr = [s.r for s in result.samples]
        srel = [s.reliability for s in result.samples]
        sc = ax3.scatter(sh, sr, c=srel, cmap="RdYlGn", vmin=0, vmax=1,
                         s=25, edgecolor="k", lw=0.3, zorder=5,
                         label="slice radius")
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="reliability")
    ax3.set_xlabel("h (along axis)")
    ax3.set_ylabel("radius")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_title("Profile r(h): points vs fitted generatrix")

    # Panel 4: per-slice reliability / coverage.
    ax4 = fig.add_subplot(2, 2, 4)
    if result.samples:
        sh = [s.h for s in result.samples]
        srel = [s.reliability for s in result.samples]
        scov = [s.angular_coverage for s in result.samples]
        ax4.plot(sh, srel, "g-o", ms=3, label="reliability")
        ax4.plot(sh, scov, "b-^", ms=3, label="angular coverage")
        ax4.axhline(result.config.min_reliability, color="r", ls="--",
                    lw=1, label="min reliability")
        ax4.set_xlabel("h (along axis)")
        ax4.set_ylabel("score")
        ax4.set_ylim(0, 1.05)
        ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("Per-slice reliability / coverage")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
