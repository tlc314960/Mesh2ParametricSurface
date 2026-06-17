"""Export the fitted proxy to mesh / polyline / JSON outputs."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from . import io_utils
from . import cross_section as cs
from .swept_tube_fitter import FitResult, ring_polylines


def _to_list(arr) -> list:
    return np.asarray(arr, dtype=float).tolist()


def _fname(base: str, prefix: str = "") -> str:
    """Build an output filename, optionally with a ``prefix_`` prefix."""
    return f"{prefix}_{base}" if prefix else base


def export_proxy(
    out_dir: str,
    part_name: str,
    source_mesh_path: str,
    result: FitResult,
    metrics: Dict,
    prefix: str = "",
) -> Dict[str, str]:
    """Write all proxy output files into ``out_dir``.

    By default filenames are clean (``proxy.ply``, ``centerline.ply``, ...)
    because the part identity is carried by the per-part directory. Pass a
    ``prefix`` to prepend ``prefix_`` if a flat layout is desired.

    Returns a dict of written paths.
    """
    io_utils.ensure_dir(out_dir)
    paths = {}

    # Swept proxy surface.
    surf_path = os.path.join(out_dir, _fname("proxy.ply", prefix))
    io_utils.save_mesh(surf_path, result.surface)
    paths["proxy"] = surf_path

    # Centerline polyline.
    cl_path = os.path.join(out_dir, _fname("centerline.ply", prefix))
    io_utils.save_polyline(cl_path, result.centerline.samples)
    paths["centerline"] = cl_path

    # Cross-section rings.
    rings = ring_polylines(result.sections, n_theta=result.config.angular_resolution)
    rings_path = os.path.join(out_dir, _fname("cross_sections.ply", prefix))
    io_utils.save_rings(rings_path, rings)
    paths["cross_sections"] = rings_path

    # JSON params.
    json_path = os.path.join(out_dir, _fname("params.json", prefix))
    payload = _build_json(part_name, source_mesh_path, result, metrics)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    paths["params"] = json_path

    return paths


def _build_json(
    part_name: str,
    source_mesh_path: str,
    result: FitResult,
    metrics: Dict,
) -> Dict:
    ep_a, ep_b = result.endpoints
    sections_json = []
    for s in result.sections:
        sections_json.append({
            "u": s.u,
            "center3d": _to_list(s.center3d),
            "tangent": _to_list(s.tangent),
            "axis_u": _to_list(s.axis_u),
            "axis_v": _to_list(s.axis_v),
            "cross_section_type": s.kind,
            "a": s.a,
            "b": s.b,
            "theta": s.theta,
            "radius": s.radius,
            "reliability": s.reliability,
            "residual": s.residual,
            "angular_coverage": s.angular_coverage,
            "n_points": s.n_points,
            "flags": s.flags,
        })

    # B-spline control points (tck = (knots, coeffs, degree)).
    tck = result.centerline.tck
    control_points = np.array(tck[1]).T.tolist()
    knots = np.array(tck[0]).tolist()
    degree = int(tck[2])

    return {
        "part_name": part_name,
        "source_mesh_path": source_mesh_path,
        "proxy_type": "generalized_cylinder",
        "cross_section_type": result.config.cross_section_kind,
        "endpoints": {
            "a": _to_list(ep_a.point),
            "b": _to_list(ep_b.point),
        },
        "centerline": {
            "sampled_points": _to_list(result.centerline.samples),
            "bspline": {
                "control_points": control_points,
                "knots": knots,
                "degree": degree,
            },
        },
        "cross_sections": sections_json,
        "fitting_metrics": {k: _json_safe(v) for k, v in metrics.items()},
        "config": {
            "n_sections": result.config.n_sections,
            "n_iters": result.config.n_iters,
            "slab_width_frac": result.config.slab_width_frac,
            "end_skip_frac": result.config.end_skip_frac,
            "cap_normal_thresh": result.config.cap_normal_thresh,
            "min_reliability": result.config.min_reliability,
            "angular_resolution": result.config.angular_resolution,
        },
    }


def _json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# --------------------------------------------------------------------------- #
# Surface-of-revolution export
# --------------------------------------------------------------------------- #
def export_revolve(
    out_dir: str,
    part_name: str,
    source_mesh_path: str,
    result,
    metrics: Dict,
    prefix: str = "",
) -> Dict[str, str]:
    """Write outputs for a surface-of-revolution fit into ``out_dir``.

    Files: proxy.ply (revolved surface), axis.ply (axis segment),
    profile.ply (the generatrix curve in 3D), params.json.
    Returns a dict of written paths.
    """
    io_utils.ensure_dir(out_dir)
    paths = {}

    surf_path = os.path.join(out_dir, _fname("proxy.ply", prefix))
    io_utils.save_mesh(surf_path, result.surface)
    paths["proxy"] = surf_path

    axis_path = os.path.join(out_dir, _fname("axis.ply", prefix))
    io_utils.save_polyline(axis_path, result.axis_segment)
    paths["axis"] = axis_path

    profile_path = os.path.join(out_dir, _fname("profile.ply", prefix))
    io_utils.save_polyline(profile_path, result.profile_polyline3d)
    paths["profile"] = profile_path

    json_path = os.path.join(out_dir, _fname("params.json", prefix))
    payload = _build_revolve_json(part_name, source_mesh_path, result, metrics)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    paths["params"] = json_path

    return paths


def _build_revolve_json(part_name, source_mesh_path, result, metrics) -> Dict:
    samples_json = []
    for s in result.samples:
        samples_json.append({
            "h": s.h,
            "r": s.r,
            "reliability": s.reliability,
            "residual": s.residual,
            "angular_coverage": s.angular_coverage,
            "n_points": s.n_points,
            "flags": s.flags,
        })
    return {
        "part_name": part_name,
        "source_mesh_path": source_mesh_path,
        "proxy_type": "surface_of_revolution",
        "axis": {
            "origin": _to_list(result.axis_origin),
            "direction": _to_list(result.axis_dir),
            "u": _to_list(result.axis_u),
            "v": _to_list(result.axis_v),
            "h_range": [float(result.h_range[0]), float(result.h_range[1])],
        },
        "profile": {
            "h": _to_list(result.profile_h),
            "r": _to_list(result.profile_r),
        },
        "profile_slices": samples_json,
        "fitting_metrics": {k: _json_safe(v) for k, v in metrics.items()},
        "config": {
            "n_profile": result.config.n_profile,
            "n_slices": result.config.n_slices,
            "n_iters": result.config.n_iters,
            "angular_resolution": result.config.angular_resolution,
            "end_skip_frac": result.config.end_skip_frac,
            "min_reliability": result.config.min_reliability,
        },
    }
