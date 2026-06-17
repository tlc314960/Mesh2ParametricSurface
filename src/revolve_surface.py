"""Surface-of-revolution (revolve) proxy fitting.

For round / squat parts (teapot body, lid, knob) the natural low-dimensional
proxy is NOT a swept tube along a curved centerline but a **surface of
revolution**: a 2D profile curve r(h) revolved about a straight axis.

Pipeline:
  1. Estimate the rotation axis (inertia-based seed + alternating refinement
     that recenters the axis through per-slice circle centers).
  2. Build the profile r(h): slice perpendicular to the axis, take a robust
     radius per slice (median distance to axis), with a reliability score.
  3. Fit a smooth profile spline through reliable (h, r) samples.
  4. Generate the revolved surface S(h, theta) = O + h*A + r(h)*(cos U + sin V).

The axis is straight, so there is no centerline curvature; the model invariant
is (axis, profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy.interpolate import UnivariateSpline


@dataclass
class RevolveConfig:
    n_profile: int = 60          # profile samples along the axis
    n_slices: int = 60           # slices used to build the profile
    n_iters: int = 4             # axis refinement iterations
    angular_resolution: int = 64
    end_skip_frac: float = 0.04  # trim extreme tips of the axis range
    min_reliability: float = 0.2
    profile_smooth: Optional[float] = None
    seed: int = 0


@dataclass
class ProfileSample:
    h: float                 # height along the axis (world units, axis origin = 0)
    r: float                 # fitted radius at this height
    reliability: float       # in [0, 1]
    residual: float          # radial scatter (RMS) within the slice
    angular_coverage: float  # fraction of 2*pi covered by slice points
    n_points: int
    flags: dict = field(default_factory=dict)


@dataclass
class RevolveResult:
    axis_origin: np.ndarray          # (3,) a point on the axis
    axis_dir: np.ndarray             # (3,) unit axis direction
    axis_u: np.ndarray               # (3,) basis vector 1 perpendicular to axis
    axis_v: np.ndarray               # (3,) basis vector 2 perpendicular to axis
    h_range: Tuple[float, float]     # (h_min, h_max) along the axis
    profile_h: np.ndarray            # (n_profile,) sampled heights
    profile_r: np.ndarray            # (n_profile,) radius at each height
    samples: List[ProfileSample]     # raw per-slice samples (diagnostics)
    surface: trimesh.Trimesh
    config: RevolveConfig
    # Convenience polylines for export / debug.
    axis_segment: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    profile_polyline3d: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


# --------------------------------------------------------------------------- #
# Axis estimation
# --------------------------------------------------------------------------- #
def _perp_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, a) * a
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(a, u)
    return u, v


def inertia_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Seed the revolution axis from the inertia tensor.

    For a surface of revolution two principal variances (the in-plane radial
    spread) are (near) equal and the third (the axis) is the odd one out. We
    return (centroid, axis_dir) where axis_dir is the eigenvector NOT belonging
    to the closest-valued pair of eigenvalues.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    # eigvals[0] <= eigvals[1] <= eigvals[2]
    gap_low = eigvals[1] - eigvals[0]   # closeness of the two smallest
    gap_high = eigvals[2] - eigvals[1]  # closeness of the two largest
    if gap_low <= gap_high:
        # Two smallest are the close pair -> axis is the largest eigenvector.
        axis = eigvecs[:, 2]
    else:
        # Two largest are the close pair -> axis is the smallest eigenvector.
        axis = eigvecs[:, 0]
    return centroid, axis / (np.linalg.norm(axis) + 1e-12)


def _axis_circularity_score(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    n_slices: int = 24,
    end_skip_frac: float = 0.05,
) -> float:
    """Score how well ``axis`` acts as a revolution axis for ``points``.

    Slices perpendicular to the axis should each form a near-complete circular
    ring (high angular coverage) with a tight radius band (low relative
    scatter). Returns the mean of (coverage * tightness) over slices.
    """
    a = axis / (np.linalg.norm(axis) + 1e-12)
    u, v = _perp_basis(a)
    rel = points - origin
    h = rel @ a
    pu = rel @ u
    pv = rel @ v
    radius = np.sqrt(pu ** 2 + pv ** 2)
    angle = np.arctan2(pv, pu)

    h_min, h_max = h.min(), h.max()
    span = h_max - h_min
    if span <= 1e-9:
        return 0.0
    lo = h_min + end_skip_frac * span
    hi = h_max - end_skip_frac * span
    edges = np.linspace(lo, hi, n_slices + 1)
    slab = 1.0 * (edges[1] - edges[0])

    scores = []
    for c_h in 0.5 * (edges[:-1] + edges[1:]):
        mask = np.abs(h - c_h) < slab
        if mask.sum() < 8:
            continue
        r_slice = radius[mask]
        r_med = float(np.median(r_slice))
        if r_med < 1e-9:
            continue
        cov = _angular_coverage(angle[mask])
        rel_res = float(np.sqrt(np.mean((r_slice - r_med) ** 2))) / r_med
        tight = np.clip(1.0 - rel_res / 0.3, 0.0, 1.0)
        scores.append(cov * tight)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def best_principal_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the revolution axis by testing all three principal directions.

    More robust than the eigenvalue-gap heuristic: the symmetry axis is the
    principal direction whose perpendicular slices are the most circular. This
    correctly handles both elongated (tall) and flat (disc / lid) revolution
    bodies, where the eigenvalue ordering of the axis differs.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    _, eigvecs = np.linalg.eigh(cov)
    best_axis = eigvecs[:, 2]
    best_score = -1.0
    for k in range(3):
        axis = eigvecs[:, k] / (np.linalg.norm(eigvecs[:, k]) + 1e-12)
        score = _axis_circularity_score(points, centroid, axis)
        if score > best_score:
            best_score = score
            best_axis = axis
    return centroid, best_axis



def _fit_circle_center_2d(pts2d: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Algebraic circle fit; returns (center(2,), radius, rms_residual)."""
    x, y = pts2d[:, 0], pts2d[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12)))
    resid = np.abs(np.linalg.norm(pts2d - [cx, cy], axis=1) - r)
    return np.array([cx, cy]), r, float(np.sqrt(np.mean(resid ** 2)))


def refine_axis(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    n_slices: int,
    n_iters: int,
    end_skip_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine the axis so it passes through per-slice circle centers.

    The centers of circular cross-sections of a surface of revolution all lie
    on the axis. We slice perpendicular to the current axis, fit a circle
    center in each slab, then refit the axis as the best-fit line through those
    centers (PCA). Repeated a few times this both recenters and re-orients the
    axis.
    """
    for _ in range(n_iters):
        u, v = _perp_basis(axis)
        h = (points - origin) @ axis
        h_min, h_max = h.min(), h.max()
        span = h_max - h_min
        lo = h_min + end_skip_frac * span
        hi = h_max - end_skip_frac * span
        edges = np.linspace(lo, hi, n_slices + 1)
        slab = 1.5 * (edges[1] - edges[0])

        centers = []
        for c_h in 0.5 * (edges[:-1] + edges[1:]):
            mask = np.abs(h - c_h) < slab
            if mask.sum() < 12:
                continue
            rel = points[mask] - origin
            pts2d = np.column_stack([rel @ u, rel @ v])
            c2d, r, rms = _fit_circle_center_2d(pts2d)
            if not np.isfinite(r) or r <= 0:
                continue
            center3d = origin + c2d[0] * u + c2d[1] * v + c_h * axis
            centers.append(center3d)
        centers = np.asarray(centers)
        if len(centers) < 3:
            break

        new_origin = centers.mean(axis=0)
        cc = centers - new_origin
        _, _, vh = np.linalg.svd(cc, full_matrices=False)
        new_axis = vh[0] / (np.linalg.norm(vh[0]) + 1e-12)
        if np.dot(new_axis, axis) < 0:
            new_axis = -new_axis
        origin, axis = new_origin, new_axis
    return origin, axis


# --------------------------------------------------------------------------- #
# Profile fitting
# --------------------------------------------------------------------------- #
def _angular_coverage(angles: np.ndarray, n_bins: int = 36) -> float:
    bins = np.floor((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(np.unique(bins)) / n_bins


def build_profile(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    cfg: RevolveConfig,
) -> Tuple[List[ProfileSample], np.ndarray, np.ndarray]:
    """Slice along the axis and build a robust radius profile r(h)."""
    u, v = _perp_basis(axis)
    rel = points - origin
    h = rel @ axis
    pu = rel @ u
    pv = rel @ v
    radius = np.sqrt(pu ** 2 + pv ** 2)
    angle = np.arctan2(pv, pu)

    h_min, h_max = h.min(), h.max()
    span = h_max - h_min
    lo = h_min + cfg.end_skip_frac * span
    hi = h_max - cfg.end_skip_frac * span
    edges = np.linspace(lo, hi, cfg.n_slices + 1)
    slab = 1.0 * (edges[1] - edges[0])

    samples: List[ProfileSample] = []
    for c_h in 0.5 * (edges[:-1] + edges[1:]):
        mask = np.abs(h - c_h) < slab
        n = int(mask.sum())
        if n < 8:
            continue
        r_slice = radius[mask]
        a_slice = angle[mask]
        # Robust radius: trimmed median (revolution => tight radius band).
        r_med = float(np.median(r_slice))
        resid = float(np.sqrt(np.mean((r_slice - r_med) ** 2)))
        cov = _angular_coverage(a_slice)
        rel_res = resid / max(r_med, 1e-9)

        flags = {}
        if cov < 0.5:
            flags["low_coverage"] = True
        if rel_res > 0.3:
            flags["high_scatter"] = True
        if r_med < 1e-4:
            flags["collapsed"] = True

        s_cov = np.clip(cov, 0.0, 1.0)
        s_res = np.clip(1.0 - rel_res / 0.3, 0.0, 1.0)
        s_count = np.clip(n / 40.0, 0.0, 1.0)
        reliability = float(s_cov * s_res * s_count)

        samples.append(ProfileSample(
            h=float(c_h), r=r_med, reliability=reliability,
            residual=resid, angular_coverage=float(cov),
            n_points=n, flags=flags,
        ))

    if len(samples) < 3:
        return samples, np.array([lo, hi]), np.array([0.0, 0.0])

    hs = np.array([s.h for s in samples])
    rs = np.array([s.r for s in samples])
    ws = np.array([max(s.reliability, 1e-3) for s in samples])

    # Smooth profile spline through (h, r), weighted by reliability.
    order = np.argsort(hs)
    hs, rs, ws = hs[order], rs[order], ws[order]
    smooth = cfg.profile_smooth
    if smooth is None:
        smooth = len(hs) * float(np.median(rs)) ** 2 * 0.05
    try:
        spl = UnivariateSpline(hs, rs, w=ws, s=smooth, k=min(3, len(hs) - 1))
        prof_h = np.linspace(hs[0], hs[-1], cfg.n_profile)
        prof_r = np.clip(spl(prof_h), 0.0, None)
    except Exception:
        prof_h = np.linspace(hs[0], hs[-1], cfg.n_profile)
        prof_r = np.interp(prof_h, hs, rs)

    return samples, prof_h, prof_r


# --------------------------------------------------------------------------- #
# Surface generation
# --------------------------------------------------------------------------- #
def generate_revolved_surface(
    origin: np.ndarray,
    axis: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    prof_h: np.ndarray,
    prof_r: np.ndarray,
    n_theta: int,
) -> trimesh.Trimesh:
    n_h = len(prof_h)
    if n_h < 2:
        return trimesh.Trimesh()
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    verts = np.zeros((n_h * n_theta, 3))
    for i in range(n_h):
        ring = (
            origin
            + prof_h[i] * axis
            + prof_r[i] * (cos_t[:, None] * u[None, :] + sin_t[:, None] * v[None, :])
        )
        verts[i * n_theta:(i + 1) * n_theta] = ring

    faces = []
    for i in range(n_h - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j2
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def profile_polyline_3d(
    origin: np.ndarray, axis: np.ndarray, u: np.ndarray,
    prof_h: np.ndarray, prof_r: np.ndarray,
) -> np.ndarray:
    """3D polyline of the profile in the U-axis plane (theta=0) for debug."""
    return origin[None, :] + prof_h[:, None] * axis[None, :] + prof_r[:, None] * u[None, :]


# --------------------------------------------------------------------------- #
# Top-level fitter
# --------------------------------------------------------------------------- #
def fit_revolve_surface(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    cfg: Optional[RevolveConfig] = None,
) -> RevolveResult:
    cfg = cfg or RevolveConfig()

    origin0, axis0 = best_principal_axis(points)
    origin, axis = refine_axis(
        points, origin0, axis0,
        n_slices=cfg.n_slices, n_iters=cfg.n_iters,
        end_skip_frac=cfg.end_skip_frac,
    )
    # Guard: for flat (disc / lid) bodies the circle centers span a tiny range
    # and the SVD re-orientation can drift. Keep the refinement only if it does
    # not reduce the perpendicular-slice circularity.
    if (_axis_circularity_score(points, origin, axis)
            < _axis_circularity_score(points, origin0, axis0) - 1e-6):
        origin, axis = origin0, axis0
    u, v = _perp_basis(axis)

    samples, prof_h, prof_r = build_profile(points, origin, axis, cfg)

    surface = generate_revolved_surface(
        origin, axis, u, v, prof_h, prof_r, cfg.angular_resolution
    )

    h = (points - origin) @ axis
    h_range = (float(h.min()), float(h.max()))
    axis_segment = np.array([origin + h_range[0] * axis, origin + h_range[1] * axis])
    prof_poly = profile_polyline_3d(origin, axis, u, prof_h, prof_r)

    return RevolveResult(
        axis_origin=origin,
        axis_dir=axis,
        axis_u=u,
        axis_v=v,
        h_range=h_range,
        profile_h=prof_h,
        profile_r=prof_r,
        samples=samples,
        surface=surface,
        config=cfg,
        axis_segment=axis_segment,
        profile_polyline3d=prof_poly,
    )
