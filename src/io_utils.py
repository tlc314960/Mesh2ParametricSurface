"""I/O helpers for loading meshes and writing debug / proxy outputs."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import trimesh


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh from disk as a single Trimesh.

    If the file contains a scene with multiple geometries, they are
    concatenated into a single mesh.
    """
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle mesh found in {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Loaded object from {path} is not a Trimesh: {type(loaded)}")
    return loaded


def save_point_cloud(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Save a point cloud as PLY.

    points : (N, 3) float
    colors : (N, 3) or (N, 4) uint8, optional.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    ensure_dir(os.path.dirname(path) or ".")
    cloud.export(path)


def save_polyline(path: str, points: np.ndarray, color: Optional[tuple] = None) -> None:
    """Save an ordered polyline as a PLY using a 3D path (edges)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        # Degenerate: store as a single point cloud so the file is still valid.
        save_point_cloud(path, points)
        return
    entities = [trimesh.path.entities.Line(points=np.arange(len(points)))]
    path3d = trimesh.path.Path3D(entities=entities, vertices=points)
    if color is not None:
        path3d.colors = np.array([color] * len(entities), dtype=np.uint8)
    ensure_dir(os.path.dirname(path) or ".")
    path3d.export(path)


def save_mesh(path: str, mesh: trimesh.Trimesh) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    mesh.export(path)


def save_rings(path: str, rings: list) -> None:
    """Save a list of cross-section rings (each (M,3)) as one combined edge path.

    Each ring is exported as a closed loop of line segments.
    """
    all_vertices = []
    entities = []
    offset = 0
    for ring in rings:
        ring = np.asarray(ring, dtype=np.float64).reshape(-1, 3)
        if len(ring) < 2:
            continue
        idx = np.arange(len(ring))
        loop = np.concatenate([idx, idx[:1]])  # close the ring
        entities.append(trimesh.path.entities.Line(points=offset + loop))
        all_vertices.append(ring)
        offset += len(ring)
    if not all_vertices:
        return
    vertices = np.concatenate(all_vertices, axis=0)
    path3d = trimesh.path.Path3D(entities=entities, vertices=vertices)
    ensure_dir(os.path.dirname(path) or ".")
    path3d.export(path)
