"""Evaluation metrics comparing the proxy surface to the wrapped input."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from . import cross_section as cs
from . import centerline as cl


def chamfer_and_coverage(
    input_points: np.ndarray,
    proxy: trimesh.Trimesh,
    n_proxy_samples: int = 20000,
    coverage_thresh_frac: float = 0.03,
    scale: float = 1.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Symmetric Chamfer distance and coverage ratio.

    coverage = fraction of input points within ``coverage_thresh_frac*scale``
    of the proxy surface.
    """
    if len(proxy.faces) == 0:
        return {
            "chamfer": float("nan"),
            "coverage_ratio": 0.0,
            "mean_input_to_proxy": float("nan"),
            "mean_proxy_to_input": float("nan"),
        }
    proxy_pts, _ = trimesh.sample.sample_surface(proxy, n_proxy_samples, seed=seed)
    proxy_pts = np.asarray(proxy_pts)

    nn_proxy = NearestNeighbors(n_neighbors=1).fit(proxy_pts)
    d_in, _ = nn_proxy.kneighbors(input_points)
    d_in = d_in.ravel()

    nn_in = NearestNeighbors(n_neighbors=1).fit(input_points)
    d_px, _ = nn_in.kneighbors(proxy_pts)
    d_px = d_px.ravel()

    thresh = coverage_thresh_frac * scale
    coverage = float(np.mean(d_in < thresh))
    chamfer = float(np.mean(d_in ** 2) + np.mean(d_px ** 2))
    return {
        "chamfer": chamfer,
        "coverage_ratio": coverage,
        "mean_input_to_proxy": float(np.mean(d_in)),
        "mean_proxy_to_input": float(np.mean(d_px)),
    }


def section_stats(
    sections: List[cs.CrossSection], min_reliability: float
) -> Dict[str, float]:
    if not sections:
        return {}
    radii = np.array([s.radius for s in sections])
    reliab = np.array([s.reliability for s in sections])
    resid = np.array([s.residual for s in sections])
    n_unreliable = int(np.sum(reliab < min_reliability))
    n_collapsed = int(np.sum([s.flags.get("collapsed", False) for s in sections]))
    return {
        "n_sections": len(sections),
        "n_unreliable": n_unreliable,
        "pct_unreliable": float(n_unreliable / len(sections)),
        "n_collapsed": n_collapsed,
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "radius_mean": float(radii.mean()),
        "avg_section_residual": float(resid.mean()),
        "mean_reliability": float(reliab.mean()),
    }


def centerline_stats(centerline: cl.Centerline) -> Dict[str, float]:
    curv = centerline.curvature(n=200)
    return {
        "centerline_length": float(centerline.arc_length()),
        "curvature_mean": float(np.mean(curv)),
        "curvature_max": float(np.max(curv)),
    }


def evaluate(
    input_points: np.ndarray,
    result,
    scale: float,
) -> Dict[str, object]:
    cfg = result.config
    metrics: Dict[str, object] = {}
    metrics.update(
        chamfer_and_coverage(input_points, result.surface, scale=scale)
    )
    metrics.update(section_stats(result.sections, cfg.min_reliability))
    metrics.update(centerline_stats(result.centerline))
    return metrics


def revolve_stats(result, input_points: np.ndarray) -> Dict[str, float]:
    """Profile / axis statistics for a surface-of-revolution fit."""
    samples = result.samples
    out: Dict[str, float] = {}
    if samples:
        rel = np.array([s.reliability for s in samples])
        resid = np.array([s.residual for s in samples])
        cov = np.array([s.angular_coverage for s in samples])
        n_unrel = int(np.sum(rel < result.config.min_reliability))
        out.update({
            "n_profile_slices": len(samples),
            "n_unreliable": n_unrel,
            "pct_unreliable": float(n_unrel / len(samples)),
            "mean_reliability": float(rel.mean()),
            "mean_slice_residual": float(resid.mean()),
            "mean_angular_coverage": float(cov.mean()),
        })
    pr = result.profile_r
    if len(pr):
        out.update({
            "radius_min": float(pr.min()),
            "radius_max": float(pr.max()),
            "radius_mean": float(pr.mean()),
        })
    out["axis_length"] = float(result.h_range[1] - result.h_range[0])

    # Roundness: how close input points are to lying on a common axis profile.
    o = result.axis_origin
    a = result.axis_dir
    rel = input_points - o
    h = rel @ a
    rad = np.linalg.norm(rel - np.outer(h, a), axis=1)
    # Compare each point's radius to the profile radius at its height.
    prof_h, prof_r = result.profile_h, result.profile_r
    if len(prof_h) >= 2:
        r_expected = np.interp(h, prof_h, prof_r,
                               left=prof_r[0], right=prof_r[-1])
        out["mean_radial_deviation"] = float(np.mean(np.abs(rad - r_expected)))
    return out


def evaluate_revolve(
    input_points: np.ndarray,
    result,
    scale: float,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics.update(
        chamfer_and_coverage(input_points, result.surface, scale=scale)
    )
    metrics.update(revolve_stats(result, input_points))
    return metrics
