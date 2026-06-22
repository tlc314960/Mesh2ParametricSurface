"""Utilities for preparing original AI-generated part meshes.

This module is intentionally separate from the alpha-wrap proxy fitting
pipeline.  The original meshes preserve visual detail for later sampling;
alpha-wrapped meshes remain the inputs to the parametric proxy fitters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh

from . import io_utils


CANONICAL_KEYWORDS = (
    ("teapot_body", ("teapot body", "teapot_body", "body")),
    ("spout", ("spout",)),
    ("handle", ("handle",)),
    ("lid", ("lid",)),
    ("knob", ("knob",)),
)


@dataclass(frozen=True)
class AlphaPart:
    index: Optional[int]
    canonical_name: str
    display_name: str
    path: Path
    mesh: trimesh.Trimesh


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_")
    return slug or "part"


def canonical_part_name(value: str) -> Optional[str]:
    """Map a semantic object/material/file name to a pipeline part name."""
    normalized = re.sub(r"[^0-9a-z]+", " ", value.lower()).strip()
    for canonical, keywords in CANONICAL_KEYWORDS:
        if any(keyword.replace("_", " ") in normalized for keyword in keywords):
            return canonical
    return None


def display_name(canonical_name: str) -> str:
    return {
        "teapot_body": "Teapot_Body",
        "spout": "Spout",
        "handle": "Handle",
        "lid": "Lid",
        "knob": "Knob",
    }.get(canonical_name, slugify(canonical_name))


def numeric_part_index(value: str) -> Optional[int]:
    match = re.search(r"(?:^|[^a-z0-9])part[_\s.-]*(\d+)(?:[^0-9]|$)", value.lower())
    return int(match.group(1)) if match else None


def discover_alpha_parts(alpha_dir: str | Path) -> List[AlphaPart]:
    """Load alpha-wrapped parts and derive their canonical names."""
    alpha_dir = Path(alpha_dir)
    parts: List[AlphaPart] = []
    for path in sorted(alpha_dir.glob("*_wrapped.obj")):
        stem = path.stem
        index = numeric_part_index(stem)
        canonical = canonical_part_name(stem)
        if canonical is None:
            canonical = slugify(re.sub(r"^part_\d+_", "", stem.replace("_wrapped", ""))).lower()
        mesh = io_utils.load_mesh(str(path))
        parts.append(
            AlphaPart(
                index=index,
                canonical_name=canonical,
                display_name=display_name(canonical),
                path=path,
                mesh=mesh,
            )
        )
    return parts


def match_alpha_part(
    object_name: str,
    material_names: Iterable[str],
    alpha_parts: Iterable[AlphaPart],
) -> Tuple[Optional[AlphaPart], str]:
    """Match an FBX mesh object to an alpha-wrapped pipeline part.

    Semantic names are preferred.  If the FBX uses generic ``part_N`` names,
    the shared numeric part index is used as a deterministic fallback.
    """
    alpha_parts = list(alpha_parts)
    semantic_sources = [object_name, *material_names]
    semantic = next(
        (canonical_part_name(value) for value in semantic_sources if canonical_part_name(value)),
        None,
    )
    if semantic is not None:
        matches = [part for part in alpha_parts if part.canonical_name == semantic]
        if len(matches) == 1:
            return matches[0], "semantic_name"

    index = numeric_part_index(object_name)
    if index is not None:
        matches = [part for part in alpha_parts if part.index == index]
        if len(matches) == 1:
            return matches[0], "numeric_index"

    return None, "unmatched"


def mesh_geometry_summary(mesh: trimesh.Trimesh) -> Dict[str, object]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    centroid = np.asarray(mesh.centroid, dtype=float)
    bbox_center = bounds.mean(axis=0)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "extents": extents.tolist(),
        "bbox_diagonal": float(np.linalg.norm(extents)),
        "centroid": centroid.tolist(),
        "bbox_center": bbox_center.tolist(),
    }


def topology_summary(mesh: trimesh.Trimesh) -> Dict[str, object]:
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edges = int(np.sum(edge_counts == 1))
    nonmanifold_edges = int(np.sum(edge_counts > 2))
    components = mesh.split(only_watertight=False)

    face_normals = np.asarray(mesh.face_normals)
    vertex_normals = np.asarray(mesh.vertex_normals)
    face_norm_lengths = np.linalg.norm(face_normals, axis=1) if len(face_normals) else np.empty(0)
    vertex_norm_lengths = (
        np.linalg.norm(vertex_normals, axis=1) if len(vertex_normals) else np.empty(0)
    )
    volume = float(mesh.volume) if len(mesh.faces) else 0.0
    return {
        "component_count": int(len(components)),
        "component_face_counts": sorted((int(len(c.faces)) for c in components), reverse=True),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "signed_volume": volume,
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "face_normals_finite": bool(np.isfinite(face_normals).all()),
        "vertex_normals_finite": bool(np.isfinite(vertex_normals).all()),
        "mean_face_normal_length": (
            float(face_norm_lengths.mean()) if len(face_norm_lengths) else None
        ),
        "mean_vertex_normal_length": (
            float(vertex_norm_lengths.mean()) if len(vertex_norm_lengths) else None
        ),
    }


def clean_original_mesh(
    mesh: trimesh.Trimesh,
    min_component_face_ratio: float = 0.0,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    """Conservatively clean an original detailed part mesh.

    No smoothing, simplification, remeshing, hole filling, or watertight repair
    is performed. Tiny components are retained by default and are removed only
    when ``min_component_face_ratio`` is explicitly greater than zero.
    """
    mesh = mesh.copy()
    before = mesh_geometry_summary(mesh)
    before_components = mesh.split(only_watertight=False)

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    removed_components: List[Dict[str, int]] = []
    if min_component_face_ratio > 0:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest_faces = max(len(component.faces) for component in components)
            threshold = max(1, int(np.ceil(min_component_face_ratio * largest_faces)))
            kept = []
            for component in components:
                if len(component.faces) >= threshold:
                    kept.append(component)
                else:
                    removed_components.append(
                        {
                            "vertex_count": int(len(component.vertices)),
                            "face_count": int(len(component.faces)),
                        }
                    )
            if kept:
                mesh = trimesh.util.concatenate(kept)

    actions: List[str] = [
        "merged_duplicate_vertices",
        "removed_degenerate_faces",
        "removed_duplicate_faces",
        "removed_unreferenced_vertices",
    ]
    if removed_components:
        actions.append("removed_tiny_components")

    winding_fixed = False
    if len(mesh.faces) and not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
        winding_fixed = bool(mesh.is_winding_consistent)
        if winding_fixed:
            actions.append("fixed_face_winding")

    inverted = False
    if mesh.is_watertight and float(mesh.volume) < 0:
        mesh.invert()
        inverted = True
        actions.append("inverted_closed_mesh_to_positive_volume")

    # Force normal computation before PLY export.
    _ = mesh.face_normals
    _ = mesh.vertex_normals

    report = {
        "before": before,
        "before_component_count": int(len(before_components)),
        "after": mesh_geometry_summary(mesh),
        "after_topology": topology_summary(mesh),
        "actions": actions,
        "removed_components": removed_components,
        "winding_fixed": winding_fixed,
        "inverted": inverted,
    }
    return mesh, report


def alignment_diagnostics(
    original: trimesh.Trimesh,
    alpha: trimesh.Trimesh,
) -> Dict[str, object]:
    """Compare original and alpha-wrap coordinates without altering either."""
    original_summary = mesh_geometry_summary(original)
    alpha_summary = mesh_geometry_summary(alpha)

    orig_bounds = np.asarray(original.bounds, dtype=float)
    alpha_bounds = np.asarray(alpha.bounds, dtype=float)
    orig_extents = orig_bounds[1] - orig_bounds[0]
    alpha_extents = alpha_bounds[1] - alpha_bounds[0]
    alpha_diag = max(float(np.linalg.norm(alpha_extents)), 1e-12)

    centroid_delta = np.asarray(original.centroid) - np.asarray(alpha.centroid)
    bbox_center_delta = orig_bounds.mean(axis=0) - alpha_bounds.mean(axis=0)
    scale_ratio = float(np.linalg.norm(orig_extents) / alpha_diag)
    extent_ratio = np.divide(
        orig_extents,
        alpha_extents,
        out=np.full(3, np.nan),
        where=np.abs(alpha_extents) > 1e-12,
    )

    center_error_fraction = float(np.linalg.norm(bbox_center_delta) / alpha_diag)
    aligned = bool(
        center_error_fraction <= 0.03
        and 0.90 <= scale_ratio <= 1.10
        and np.all((extent_ratio >= 0.75) & (extent_ratio <= 1.25))
    )
    warnings = []
    if not 0.90 <= scale_ratio <= 1.10:
        warnings.append("possible_global_scale_mismatch")
    if center_error_fraction > 0.03:
        warnings.append("possible_translation_or_transform_mismatch")
    if not np.all((extent_ratio >= 0.75) & (extent_ratio <= 1.25)):
        warnings.append("possible_axis_swap_or_per_axis_scale_mismatch")

    return {
        "original": original_summary,
        "alpha_wrapped": alpha_summary,
        "centroid_difference": centroid_delta.tolist(),
        "centroid_difference_norm": float(np.linalg.norm(centroid_delta)),
        "bbox_center_difference": bbox_center_delta.tolist(),
        "bbox_center_difference_norm": float(np.linalg.norm(bbox_center_delta)),
        "bbox_center_error_fraction": center_error_fraction,
        "approximate_scale_ratio": scale_ratio,
        "per_axis_extent_ratio": extent_ratio.tolist(),
        "coordinate_systems_appear_aligned": aligned,
        "warnings": warnings,
    }


def save_alignment_figure(
    out_path: str | Path,
    matched_parts: Iterable[Tuple[str, trimesh.Trimesh, trimesh.Trimesh]],
    max_points_per_mesh: int = 2500,
) -> None:
    """Save a compact original-vs-alpha alignment overview."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched_parts = list(matched_parts)
    if not matched_parts:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(matched_parts) / n_cols))
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    rng = np.random.default_rng(0)

    for index, (name, original, alpha) in enumerate(matched_parts, start=1):
        ax = fig.add_subplot(n_rows, n_cols, index, projection="3d")
        for mesh, color, label, marker_size in (
            (original, "steelblue", "original FBX", 1.2),
            (alpha, "darkorange", "alpha wrap", 4.0),
        ):
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > max_points_per_mesh:
                sample = rng.choice(len(vertices), max_points_per_mesh, replace=False)
                vertices = vertices[sample]
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                s=marker_size,
                c=color,
                alpha=0.45,
                label=label,
            )

        combined = np.vstack([original.bounds, alpha.bounds])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(float((maxs - mins).max()) * 0.5, 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Original AI meshes vs alpha-wrapped proxy inputs")
    fig.tight_layout()
    io_utils.ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
