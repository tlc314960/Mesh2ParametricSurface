"""Surface sampling utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


def sample_surface(
    mesh: trimesh.Trimesh,
    n_points: int = 20000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points and per-point normals on the mesh surface.

    Returns
    -------
    points : (N, 3) float
    normals : (N, 3) float (unit length)
    """
    rng = np.random.default_rng(seed)
    # trimesh's sample_surface uses the global numpy RNG via face_weight; we
    # pass an explicit seed by temporarily seeding numpy for reproducibility.
    points, face_idx = trimesh.sample.sample_surface(
        mesh, n_points, seed=seed
    )
    normals = mesh.face_normals[face_idx]
    # Normalize defensively.
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)
