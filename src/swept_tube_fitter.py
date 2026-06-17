"""Swept-tube (generalized cylinder) fitter with alternating refinement.

Brings together endpoint estimation, centerline initialization, robust
cross-section fitting, and the alternating centerline/cross-section refinement
loop. Finally generates a swept outer surface using parallel-transport frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import trimesh

from . import centerline as cl
from . import cross_section as cs
from .endpoint_estimation import estimate_endpoints, Endpoint


@dataclass
class FitConfig:
    n_sections: int = 40
    n_iters: int = 4
    cross_section_kind: str = "circle"  # 'circle' or 'ellipse'
    slab_width_frac: float = 0.04       # fraction of centerline length
    end_skip_frac: float = 0.08         # skip/downweight first & last fraction
    cap_normal_thresh: float = 0.7
    min_reliability: float = 0.15
    angular_resolution: int = 48        # samples around each ring for surface
    spline_smooth: Optional[float] = None
    # Adaptive radial gating: a slab only keeps points within
    # ``radius_gate_factor * running_radius`` of the centerline, so a cutting
    # plane through a tightly-curved tube (e.g. a teapot handle) cannot grab
    # the opposite arm. ``max_radius_factor`` flags / downweights cross-sections
    # whose fitted radius blows past the running tube radius.
    radius_gate_factor: float = 2.0
    max_radius_factor: float = 1.7
    seed: int = 0


@dataclass
class FitResult:
    centerline: cl.Centerline
    initial_centerline: np.ndarray
    sections: List[cs.CrossSection]
    endpoints: tuple
    surface: trimesh.Trimesh
    config: FitConfig
    radii_profile: np.ndarray = field(default_factory=lambda: np.empty(0))


def _end_penalty(u: float, end_skip_frac: float) -> float:
    """Ramp penalty: 1 at the very ends, 0 in the interior."""
    if u < end_skip_frac:
        return 1.0 - u / max(end_skip_frac, 1e-9)
    if u > 1.0 - end_skip_frac:
        return 1.0 - (1.0 - u) / max(end_skip_frac, 1e-9)
    return 0.0


def estimate_tube_radius(points: np.ndarray, centerline: cl.Centerline,
                         n_samples: int = 200) -> float:
    """Robust estimate of the tube radius.

    For a generalized cylinder every surface point sits ~r from the centerline,
    so the median distance from the sampled surface points to the centerline is
    a good radius estimate that is robust to the tube being curved (each point's
    nearest centerline sample lies on its own portion of the tube). This does
    not depend on cross-section fitting, so it gives a stable bootstrap value
    even before the first slicing pass.
    """
    from scipy.spatial import cKDTree

    cl_pts = centerline.position(np.linspace(0.0, 1.0, n_samples))
    tree = cKDTree(cl_pts)
    d, _ = tree.query(points)
    return float(np.median(d))


def _fit_sections_along(
    centerline: cl.Centerline,
    points: np.ndarray,
    normals: Optional[np.ndarray],
    cfg: FitConfig,
    total_length: float,
    running_radius: float,
) -> List[cs.CrossSection]:
    us = np.linspace(0.0, 1.0, cfg.n_sections)
    centers = centerline.position(us)
    tangents = centerline.tangent(us)
    slab_width = cfg.slab_width_frac * total_length
    # Adaptive radial gate: never grab points farther than a few tube radii
    # from the centerline (prevents catching the opposite arm of a tight C).
    radius_limit = cfg.radius_gate_factor * running_radius
    max_radius = cfg.max_radius_factor * running_radius

    sections: List[cs.CrossSection] = []
    for i, u in enumerate(us):
        sel_pts, sel_norms = cs.select_slab_points(
            points, normals, centers[i], tangents[i],
            slab_width=slab_width,
            cap_normal_thresh=cfg.cap_normal_thresh,
            radius_limit=radius_limit,
        )
        if len(sel_pts) < 6:
            continue
        sec = cs.fit_cross_section(
            u=u,
            center_guess=centers[i],
            tangent=tangents[i],
            sel_pts=sel_pts,
            kind=cfg.cross_section_kind,
            end_penalty=_end_penalty(u, cfg.end_skip_frac),
            max_radius=max_radius,
            seed=cfg.seed + i,
        )
        if sec is not None:
            sections.append(sec)
    return sections


def fit_swept_tube(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    cfg: Optional[FitConfig] = None,
) -> FitResult:
    cfg = cfg or FitConfig()

    # 1. Endpoints + surface graph.
    ep_a, ep_b, graph = estimate_endpoints(points)

    # 2. Initial centerline (coarse, used only for initialization).
    init_pts = cl.initial_centerline_from_graph(
        points, graph, ep_a.index, ep_b.index, n_samples=cfg.n_sections
    )
    centerline = cl.fit_spline_centerline(
        init_pts,
        weights=None,
        smooth=cfg.spline_smooth,
        n_samples=max(cfg.n_sections, 60),
    )
    total_length = centerline.arc_length()

    # Bootstrap tube radius from point-to-centerline distances (robust, does
    # not require cross-section fitting). Used to gate the slab radius so a
    # tightly curved tube (handle) cannot grab its opposite arm.
    running_radius = estimate_tube_radius(points, centerline)

    sections: List[cs.CrossSection] = []
    # 3. Alternating refinement.
    for it in range(cfg.n_iters):
        sections = _fit_sections_along(
            centerline, points, normals, cfg, total_length, running_radius
        )
        reliable = [s for s in sections if s.reliability >= cfg.min_reliability]
        if len(reliable) < 4:
            # Relax threshold once if too few survive.
            reliable = sorted(sections, key=lambda s: -s.reliability)[: max(4, len(sections) // 2)]
        if len(reliable) < 4:
            break

        reliable.sort(key=lambda s: s.u)
        centers = np.array([s.center3d for s in reliable])
        weights = np.array([s.reliability for s in reliable])

        new_centerline = cl.fit_spline_centerline(
            centers,
            weights=weights,
            smooth=cfg.spline_smooth,
            n_samples=max(cfg.n_sections, 60),
        )
        centerline = new_centerline
        total_length = centerline.arc_length()
        # Update the running radius from the refined centerline + reliable fits.
        rel_radii = np.array([s.radius for s in reliable if s.radius > 1e-6])
        geom_radius = estimate_tube_radius(points, centerline)
        if len(rel_radii):
            # Blend the geometric estimate with the median fitted radius.
            running_radius = float(np.median([geom_radius, np.median(rel_radii)]))
        else:
            running_radius = geom_radius

    # 4. Final cross-section pass on the converged centerline.
    sections = _fit_sections_along(
        centerline, points, normals, cfg, total_length, running_radius
    )
    sections.sort(key=lambda s: s.u)

    # 5. Generate swept surface.
    surface, radii = generate_swept_surface(centerline, sections, cfg)

    return FitResult(
        centerline=centerline,
        initial_centerline=init_pts,
        sections=sections,
        endpoints=(ep_a, ep_b),
        surface=surface,
        config=cfg,
        radii_profile=radii,
    )


def _interp_section_params(
    sections: List[cs.CrossSection],
    us: np.ndarray,
    min_reliability: float,
):
    """Interpolate (a, b, theta) along the tube from reliable sections."""
    reliable = [s for s in sections if s.reliability >= min_reliability]
    if len(reliable) < 2:
        reliable = sorted(sections, key=lambda s: -s.reliability)[: max(2, len(sections) // 2)]
    reliable.sort(key=lambda s: s.u)

    su = np.array([s.u for s in reliable])
    sa = np.array([s.a for s in reliable])
    sb = np.array([s.b for s in reliable])
    # Unwrap theta for smooth interpolation.
    st = np.unwrap(np.array([s.theta for s in reliable]) * 2.0) / 2.0

    a = np.interp(us, su, sa)
    b = np.interp(us, su, sb)
    theta = np.interp(us, su, st)
    return a, b, theta


def generate_swept_surface(
    centerline: cl.Centerline,
    sections: List[cs.CrossSection],
    cfg: FitConfig,
):
    """Generate an open (un-capped) swept tube surface, periodic in angle."""
    n_u = max(cfg.n_sections, 60)
    us = np.linspace(0.0, 1.0, n_u)
    centers, U, V = centerline.parallel_transport_frames(us)

    if len(sections) == 0:
        empty = trimesh.Trimesh()
        return empty, np.empty(0)

    a, b, theta = _interp_section_params(sections, us, cfg.min_reliability)
    radii = 0.5 * (a + b)

    n_theta = cfg.angular_resolution
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    verts = np.zeros((n_u * n_theta, 3))
    for i in range(n_u):
        ct, st = np.cos(theta[i]), np.sin(theta[i])
        # Rotate the local frame by in-plane ellipse angle theta.
        Ui = ct * U[i] + st * V[i]
        Vi = -st * U[i] + ct * V[i]
        ring = (
            centers[i]
            + a[i] * np.cos(angles)[:, None] * Ui[None, :]
            + b[i] * np.sin(angles)[:, None] * Vi[None, :]
        )
        verts[i * n_theta:(i + 1) * n_theta] = ring

    faces = []
    for i in range(n_u - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j2
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int64)

    surface = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return surface, radii


def ring_polylines(
    sections: List[cs.CrossSection], n_theta: int = 48
) -> List[np.ndarray]:
    """Build 3D polylines for each fitted cross-section ring (for debug)."""
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rings = []
    for s in sections:
        ct, st = np.cos(s.theta), np.sin(s.theta)
        Ui = ct * s.axis_u + st * s.axis_v
        Vi = -st * s.axis_u + ct * s.axis_v
        ring = (
            s.center3d
            + s.a * np.cos(angles)[:, None] * Ui[None, :]
            + s.b * np.sin(angles)[:, None] * Vi[None, :]
        )
        rings.append(ring)
    return rings
