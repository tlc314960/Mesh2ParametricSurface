"""Centerline representation, initialization and smooth refitting.

The centerline is stored as an ordered set of sample points with an arc-length
parameter ``s``. We provide:

  * ``initial_centerline_from_graph``: a coarse ordering using a shortest path
    between the two endpoint regions over the kNN surface graph. The surface
    path is pulled toward the interior (averaged with nearby points) and
    smoothed so it is used only as initialization, NOT as the final result.
  * ``fit_spline_centerline``: a weighted smoothing cubic B-spline through a
    set of (reliable) center points, with curvature regularization via the
    smoothing factor.
  * ``Centerline``: a sampleable object exposing position and unit tangent at
    arbitrary arc-length, plus parallel-transport local frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.sparse.csgraph import dijkstra


def _arc_length(points: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points uniformly in arc length."""
    points = np.asarray(points, dtype=np.float64)
    s = _arc_length(points)
    if s[-1] <= 0:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0, s[-1], n)
    out = np.empty((n, 3))
    for d in range(3):
        out[:, d] = np.interp(target, s, points[:, d])
    return out


def initial_centerline_from_graph(
    points: np.ndarray,
    graph,
    idx_a: int,
    idx_b: int,
    n_samples: int = 40,
    interior_k: int = 25,
    smooth_iters: int = 8,
) -> np.ndarray:
    """Coarse centerline initialization.

    1. Shortest path on the surface kNN graph between the two endpoints.
    2. Pull each path vertex toward the local interior by averaging its
       ``interior_k`` nearest neighbours (this moves the path off the surface
       toward the tube center).
    3. Laplacian smoothing + uniform arc-length resampling.
    """
    from sklearn.neighbors import NearestNeighbors

    dist, predecessors = dijkstra(
        graph, indices=idx_a, directed=False, return_predecessors=True
    )
    # Reconstruct path from idx_b back to idx_a.
    path = []
    cur = idx_b
    guard = 0
    while cur != idx_a and cur >= 0 and guard < len(points) + 5:
        path.append(cur)
        cur = predecessors[cur]
        guard += 1
    path.append(idx_a)
    path = np.array(path[::-1], dtype=int)

    if len(path) < 2:
        # Fallback: straight line between the two endpoints.
        return resample_polyline(points[[idx_a, idx_b]], n_samples)

    path_pts = points[path]

    # Pull toward interior using neighbours in the full cloud.
    nn = NearestNeighbors(n_neighbors=min(interior_k, len(points))).fit(points)
    _, nbr_idx = nn.kneighbors(path_pts)
    interior_pts = points[nbr_idx].mean(axis=1)

    # Laplacian smoothing of the interior polyline (keep endpoints fixed-ish).
    smoothed = interior_pts.copy()
    for _ in range(smooth_iters):
        prev = smoothed.copy()
        smoothed[1:-1] = 0.5 * prev[1:-1] + 0.25 * (prev[:-2] + prev[2:])

    return resample_polyline(smoothed, n_samples)


def fit_spline_centerline(
    centers: np.ndarray,
    weights: Optional[np.ndarray] = None,
    smooth: Optional[float] = None,
    n_samples: int = 60,
    degree: int = 3,
) -> "Centerline":
    """Fit a weighted smoothing cubic B-spline through center points.

    ``centers`` should be ordered along the tube. ``weights`` (per point) come
    from cross-section reliability. ``smooth`` is the spline smoothing factor
    ``s`` passed to :func:`scipy.interpolate.splprep`; larger means smoother.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    k = min(degree, max(1, n - 1))

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64).copy()
        w[w <= 0] = 1e-3

    if smooth is None:
        # Heuristic: scale with point count so the spline follows the data but
        # filters high-frequency noise.
        smooth = float(n) * 0.5

    # splprep needs a parameterization; use normalized arc length.
    u = _arc_length(centers)
    if u[-1] <= 0:
        u = np.linspace(0, 1, n)
    else:
        u = u / u[-1]

    try:
        tck, _ = splprep(centers.T, w=w, u=u, s=smooth, k=k)
    except Exception:
        tck, _ = splprep(centers.T, u=u, s=smooth, k=k)

    uu = np.linspace(0, 1, n_samples)
    pts = np.array(splev(uu, tck)).T
    return Centerline(tck=tck, samples=pts)


@dataclass
class Centerline:
    tck: tuple
    samples: np.ndarray  # (n_samples, 3) cached uniform samples

    def position(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        return np.array(splev(np.clip(u, 0, 1), self.tck)).T

    def tangent(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        der = np.array(splev(np.clip(u, 0, 1), self.tck, der=1)).T
        norm = np.linalg.norm(der, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return der / norm

    def arc_length(self, n: int = 200) -> float:
        uu = np.linspace(0, 1, n)
        pts = self.position(uu)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def curvature(self, n: int = 200) -> np.ndarray:
        uu = np.linspace(0, 1, n)
        d1 = np.array(splev(uu, self.tck, der=1)).T
        d2 = np.array(splev(uu, self.tck, der=2)).T
        cross = np.cross(d1, d2)
        num = np.linalg.norm(cross, axis=1)
        den = np.linalg.norm(d1, axis=1) ** 3 + 1e-12
        return num / den

    def parallel_transport_frames(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (centers, U, V) frames at parameters ``u`` using parallel
        transport to avoid twisting / Frenet instability.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        centers = self.position(u)
        tangents = self.tangent(u)

        n = len(u)
        U = np.zeros((n, 3))
        V = np.zeros((n, 3))

        # Initialize the first normal: any vector perpendicular to tangent[0].
        t0 = tangents[0]
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(t0, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u0 = ref - np.dot(ref, t0) * t0
        u0 /= np.linalg.norm(u0) + 1e-12
        U[0] = u0
        V[0] = np.cross(t0, u0)

        for i in range(1, n):
            t_prev = tangents[i - 1]
            t_cur = tangents[i]
            v = np.cross(t_prev, t_cur)
            s = np.linalg.norm(v)
            c = np.dot(t_prev, t_cur)
            u_prev = U[i - 1]
            if s < 1e-8:
                u_rot = u_prev
            else:
                axis = v / s
                # Rotate u_prev by angle between tangents about axis (Rodrigues).
                angle = np.arctan2(s, c)
                u_rot = (
                    u_prev * np.cos(angle)
                    + np.cross(axis, u_prev) * np.sin(angle)
                    + axis * np.dot(axis, u_prev) * (1 - np.cos(angle))
                )
            # Re-orthogonalize against current tangent.
            u_rot = u_rot - np.dot(u_rot, t_cur) * t_cur
            u_rot /= np.linalg.norm(u_rot) + 1e-12
            U[i] = u_rot
            V[i] = np.cross(t_cur, u_rot)
        return centers, U, V
