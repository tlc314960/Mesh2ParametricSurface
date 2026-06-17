"""Robust cross-section slicing and circle / ellipse fitting.

For each position along the centerline we:
  * gather points in a slab perpendicular to the local tangent,
  * optionally reject cap-like / end-facing points using normals,
  * project them into the local 2D slicing plane,
  * robustly fit a circle or ellipse (RANSAC + residual trimming),
  * convert the 2D center back to 3D (a candidate centerline point),
  * compute a reliability score.

The reliability score is the most important diagnostic: it combines point
count, angular coverage, residual error, aspect ratio / radius sanity and a
penalty for sections close to the endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CrossSection:
    u: float                      # centerline parameter in [0, 1]
    center3d: np.ndarray          # (3,) fitted center in world space
    tangent: np.ndarray           # (3,) local tangent used for slicing
    axis_u: np.ndarray            # (3,) plane basis vector 1
    axis_v: np.ndarray            # (3,) plane basis vector 2
    kind: str                     # 'circle' or 'ellipse'
    radius: float                 # circle radius or mean ellipse radius
    a: float                      # ellipse semi-axis along axis_u (==radius for circle)
    b: float                      # ellipse semi-axis along axis_v (==radius for circle)
    theta: float                  # ellipse rotation in-plane (radians)
    reliability: float            # in [0, 1]
    residual: float               # RMS radial residual (world units)
    angular_coverage: float       # fraction of 2*pi covered by inliers
    n_points: int                 # number of points used (inliers)
    flags: dict = field(default_factory=dict)


def _plane_basis(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, t) * t
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(t, u)
    return u, v


def select_slab_points(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    center: np.ndarray,
    tangent: np.ndarray,
    slab_width: float,
    cap_normal_thresh: float = 0.7,
    radius_limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select points within a slab perpendicular to ``tangent``.

    Returns the selected (points, normals_or_None). Points whose normal is
    nearly parallel to the tangent (cap-like / end-facing) are filtered when
    normals are available.
    """
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    rel = points - center
    along = rel @ t
    in_slab = np.abs(along) < slab_width

    if radius_limit is not None:
        perp = rel - np.outer(along, t)
        radial = np.linalg.norm(perp, axis=1)
        in_slab &= radial < radius_limit

    idx = np.nonzero(in_slab)[0]
    if normals is not None and len(idx) > 0:
        n_dot = np.abs(normals[idx] @ t)
        side_wall = n_dot < cap_normal_thresh
        if side_wall.sum() >= 8:  # only filter if enough side-wall points remain
            idx = idx[side_wall]

    sel_pts = points[idx]
    sel_norms = normals[idx] if normals is not None else None
    return sel_pts, sel_norms


# --------------------------------------------------------------------------- #
# 2D robust circle / ellipse fitting
# --------------------------------------------------------------------------- #
def fit_circle_2d(pts2d: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit. Returns (center(2,), radius)."""
    x = pts2d[:, 0]
    y = pts2d[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12))
    return np.array([cx, cy]), float(r)


def fit_circle_ransac(
    pts2d: np.ndarray,
    iters: int = 200,
    inlier_frac_trim: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """RANSAC circle fit with final residual trimming.

    Returns (center(2,), radius, inlier_mask).
    """
    n = len(pts2d)
    rng = np.random.default_rng(seed)
    if n < 3:
        c, r = (pts2d.mean(axis=0), 0.0) if n else (np.zeros(2), 0.0)
        return c, r, np.ones(n, dtype=bool)

    best_inliers = None
    best_score = -1
    # Scale-aware threshold from a coarse fit.
    c0, r0 = fit_circle_2d(pts2d)
    thresh = max(0.08 * r0, 1e-6)

    for _ in range(iters):
        sample = pts2d[rng.choice(n, 3, replace=False)]
        try:
            c, r = fit_circle_2d(sample)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(r) or r <= 0:
            continue
        d = np.abs(np.linalg.norm(pts2d - c, axis=1) - r)
        inliers = d < thresh
        score = inliers.sum()
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 3:
        c, r = fit_circle_2d(pts2d)
        return c, r, np.ones(n, dtype=bool)

    # Refit on inliers, then trim worst residuals once more (IRLS-lite).
    inl = best_inliers.copy()
    for _ in range(3):
        c, r = fit_circle_2d(pts2d[inl])
        d = np.abs(np.linalg.norm(pts2d - c, axis=1) - r)
        keep_thresh = np.quantile(d[inl], 1.0 - inlier_frac_trim)
        new_inl = d <= max(keep_thresh, thresh)
        if new_inl.sum() < 3 or np.array_equal(new_inl, inl):
            inl = new_inl if new_inl.sum() >= 3 else inl
            break
        inl = new_inl
    c, r = fit_circle_2d(pts2d[inl])
    return c, float(r), inl


def fit_ellipse_2d(pts2d: np.ndarray) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """Direct least-squares ellipse fit (Fitzgibbon).

    Returns (center(2,), a, b, theta) or None if degenerate.
    """
    x = pts2d[:, 0]
    y = pts2d[:, 1]
    D = np.column_stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return None
    # The ellipse solution corresponds to the positive eigenvalue.
    valid = np.isfinite(eigvals) & (eigvals > 0)
    if not np.any(valid):
        return None
    a_vec = eigvecs[:, valid][:, np.argmax(eigvals[valid])].real
    A, B, Cc, Dd, E, F = a_vec
    denom = B ** 2 - 4 * A * Cc
    if abs(denom) < 1e-12:
        return None
    cx = (2 * Cc * Dd - B * E) / denom
    cy = (2 * A * E - B * Dd) / denom

    num = 2 * (A * E ** 2 + Cc * Dd ** 2 + F * B ** 2 - B * Dd * E - 4 * A * Cc * F)
    s = np.sqrt((A - Cc) ** 2 + B ** 2)
    ax1_den = denom * ((A + Cc) + s)
    ax2_den = denom * ((A + Cc) - s)
    if ax1_den == 0 or ax2_den == 0:
        return None
    ax1_sq = num / ax1_den
    ax2_sq = num / ax2_den
    if ax1_sq <= 0 or ax2_sq <= 0:
        return None
    ax1 = np.sqrt(ax1_sq)
    ax2 = np.sqrt(ax2_sq)
    if abs(B) < 1e-12:
        theta = 0.0 if A < Cc else np.pi / 2
    else:
        theta = 0.5 * np.arctan2(B, (A - Cc))
    a = max(ax1, ax2)
    b = min(ax1, ax2)
    return np.array([cx, cy]), float(a), float(b), float(theta)


def _angular_coverage(pts2d: np.ndarray, center: np.ndarray, n_bins: int = 36) -> float:
    ang = np.arctan2(pts2d[:, 1] - center[1], pts2d[:, 0] - center[0])
    bins = np.floor((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(np.unique(bins)) / n_bins


def fit_cross_section(
    u: float,
    center_guess: np.ndarray,
    tangent: np.ndarray,
    sel_pts: np.ndarray,
    kind: str = "circle",
    end_penalty: float = 0.0,
    max_radius: Optional[float] = None,
    seed: int = 0,
) -> Optional[CrossSection]:
    """Fit a robust cross-section to slab points.

    ``end_penalty`` in [0,1] downweights sections near the endpoints.
    Returns None if there are too few points to fit anything.
    """
    if len(sel_pts) < 6:
        return None

    axis_u, axis_v = _plane_basis(tangent)
    rel = sel_pts - center_guess
    pts2d = np.column_stack([rel @ axis_u, rel @ axis_v])

    c2d, r, inl = fit_circle_ransac(pts2d, seed=seed)
    a = b = r
    theta = 0.0
    used_kind = "circle"

    if kind == "ellipse" and inl.sum() >= 8:
        ell = fit_ellipse_2d(pts2d[inl])
        if ell is not None:
            ce, ea, eb, etheta = ell
            # Sanity: reject wildly eccentric or huge ellipses.
            radius_ok = max_radius is None or ea < max_radius
            if eb > 1e-6 and ea / eb < 4.0 and ea < 3.0 * max(r, 1e-6) and radius_ok:
                c2d, a, b, theta = ce, ea, eb, etheta
                r = 0.5 * (a + b)
                used_kind = "ellipse"

    center3d = center_guess + c2d[0] * axis_u + c2d[1] * axis_v

    # Residuals (radial) on inliers in world scale.
    inl_pts = pts2d[inl]
    if used_kind == "circle":
        resid = np.abs(np.linalg.norm(inl_pts - c2d, axis=1) - r)
    else:
        # Approximate ellipse residual via normalized radial distance.
        d = inl_pts - c2d
        ct, st = np.cos(-theta), np.sin(-theta)
        xr = d[:, 0] * ct - d[:, 1] * st
        yr = d[:, 0] * st + d[:, 1] * ct
        norm_r = np.sqrt((xr / max(a, 1e-9)) ** 2 + (yr / max(b, 1e-9)) ** 2)
        resid = np.abs(norm_r - 1.0) * r
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 1e9

    coverage = _angular_coverage(inl_pts, c2d)
    n_inl = int(inl.sum())

    reliability, flags = _reliability(
        n_inl, coverage, rms, r, a, b, used_kind, end_penalty, max_radius
    )

    return CrossSection(
        u=float(u),
        center3d=center3d,
        tangent=tangent / (np.linalg.norm(tangent) + 1e-12),
        axis_u=axis_u,
        axis_v=axis_v,
        kind=used_kind,
        radius=float(r),
        a=float(a),
        b=float(b),
        theta=float(theta),
        reliability=float(reliability),
        residual=rms,
        angular_coverage=float(coverage),
        n_points=n_inl,
        flags=flags,
    )


def _reliability(
    n_inl: int,
    coverage: float,
    rms: float,
    r: float,
    a: float,
    b: float,
    kind: str,
    end_penalty: float,
    max_radius: Optional[float] = None,
) -> Tuple[float, dict]:
    flags = {}
    # Point-count score (saturates ~40 points).
    s_count = np.clip(n_inl / 40.0, 0.0, 1.0)
    # Angular coverage score (full ring => 1).
    s_cov = np.clip(coverage, 0.0, 1.0)
    # Residual score relative to radius.
    rel_res = rms / max(r, 1e-9)
    s_res = np.clip(1.0 - rel_res / 0.25, 0.0, 1.0)
    # Aspect ratio score for ellipse (penalize very eccentric).
    aspect = a / max(b, 1e-9)
    s_aspect = np.clip(1.0 - (aspect - 1.0) / 3.0, 0.0, 1.0)
    # Oversized score: a fitted radius far above the running tube radius means
    # the slab grabbed unrelated geometry (e.g. the opposite arm of a handle).
    s_size = 1.0
    oversized = False
    if max_radius is not None and max_radius > 1e-9:
        ratio = r / max_radius
        if ratio > 1.0:
            oversized = True
            s_size = float(np.clip(1.0 - (ratio - 1.0), 0.0, 1.0))

    collapsed = r < 1e-4
    if collapsed:
        flags["collapsed"] = True
    if coverage < 0.4:
        flags["low_coverage"] = True
    if rel_res > 0.25:
        flags["high_residual"] = True
    if oversized:
        flags["oversized"] = True

    score = s_count * s_cov * s_res * s_aspect * s_size
    score *= (1.0 - np.clip(end_penalty, 0.0, 1.0))
    if collapsed:
        score = 0.0
    return float(score), flags
