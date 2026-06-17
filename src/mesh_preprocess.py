"""Mesh preprocessing: cleanup, component filtering, degenerate removal."""

from __future__ import annotations

import numpy as np
import trimesh


def clean_mesh(
    mesh: trimesh.Trimesh,
    min_component_face_ratio: float = 0.05,
) -> trimesh.Trimesh:
    """Clean a noisy alpha-wrapped mesh.

    Steps:
      * merge duplicate vertices
      * remove degenerate / duplicate faces
      * drop tiny disconnected components (relative to the largest one)

    Parameters
    ----------
    min_component_face_ratio : float
        Connected components whose face count is below this fraction of the
        largest component's face count are discarded.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        face_counts = np.array([len(c.faces) for c in components])
        keep_thresh = max(1, int(min_component_face_ratio * face_counts.max()))
        kept = [c for c, n in zip(components, face_counts) if n >= keep_thresh]
        if kept:
            mesh = trimesh.util.concatenate(kept)

    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def mesh_scale(mesh: trimesh.Trimesh) -> float:
    """A robust characteristic length scale (diagonal of the bounding box)."""
    extents = mesh.bounds[1] - mesh.bounds[0]
    return float(np.linalg.norm(extents))
