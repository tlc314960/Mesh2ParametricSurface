"""Geometric endpoint estimation for elongated (spout-like) envelopes.

The alpha-wrapped mesh may be watertight and seal the spout ends, so we do
NOT use boundary loops. Instead we estimate the two ends geometrically:

  1. PCA gives a coarse major axis to orient the part.
  2. Candidate extremal points are taken from both ends along that axis.
  3. A kNN graph over the sampled points is used to refine the pair by
     maximizing the graph (geodesic-ish) distance between the candidate
     regions, which follows the curved surface rather than cutting through
     space.

We return endpoint *regions* (a representative point plus the indices of
nearby points) rather than a single noisy vertex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors


@dataclass
class Endpoint:
    point: np.ndarray            # (3,) representative location of the end
    index: int                   # index of the seed point in the cloud
    member_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroid, principal_axis) of a point cloud."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return centroid, axis / (np.linalg.norm(axis) + 1e-12)


def build_knn_graph(points: np.ndarray, k: int = 12) -> csr_matrix:
    """Build a symmetric kNN graph weighted by Euclidean distance.

    The graph is connected by progressively increasing k if necessary so that
    geodesic distances are well-defined across the whole cloud.
    """
    n = len(points)
    k = min(max(k, 2), n - 1)
    while True:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
        dist, idx = nn.kneighbors(points)
        rows = np.repeat(np.arange(n), k)
        cols = idx[:, 1:].reshape(-1)
        vals = dist[:, 1:].reshape(-1)
        graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
        graph = graph.maximum(graph.T)  # symmetrize
        n_comp, _ = connected_components(graph, directed=False)
        if n_comp == 1 or k >= n - 1:
            return graph
        k = min(n - 1, k * 2)


def _region_indices(points: np.ndarray, seed_idx: int, radius: float) -> np.ndarray:
    d = np.linalg.norm(points - points[seed_idx], axis=1)
    return np.nonzero(d < radius)[0]


def estimate_endpoints(
    points: np.ndarray,
    k: int = 12,
    region_frac: float = 0.06,
    n_axis_candidates: int = 30,
) -> Tuple[Endpoint, Endpoint, np.ndarray]:
    """Estimate the two endpoints of an elongated envelope.

    Returns
    -------
    ep_a, ep_b : Endpoint
    graph : csr_matrix
        The kNN graph (reused by downstream centerline initialization).
    """
    n = len(points)
    centroid, axis = pca_axis(points)
    proj = (points - centroid) @ axis

    # Candidate pools at both extremes along the PCA axis.
    order = np.argsort(proj)
    low_pool = order[:n_axis_candidates]
    high_pool = order[-n_axis_candidates:]

    graph = build_knn_graph(points, k=k)

    # Among candidate pairs (low, high), pick the pair with the largest
    # geodesic distance. To keep it cheap, run dijkstra from each low
    # candidate seed is expensive; instead run from the single most extreme
    # low point and the most extreme high point, then refine.
    src_low = low_pool[0]
    src_high = high_pool[-1]

    dist_from_low = dijkstra(graph, indices=src_low, directed=False)
    dist_from_high = dijkstra(graph, indices=src_high, directed=False)

    # Endpoint A: farthest reachable point from src_high (true geodesic tip).
    finite_low = np.where(np.isfinite(dist_from_high), dist_from_high, -np.inf)
    idx_a = int(np.argmax(finite_low))
    # Endpoint B: farthest reachable point from idx_a.
    dist_from_a = dijkstra(graph, indices=idx_a, directed=False)
    finite_a = np.where(np.isfinite(dist_from_a), dist_from_a, -np.inf)
    idx_b = int(np.argmax(finite_a))

    scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    radius = region_frac * scale
    mem_a = _region_indices(points, idx_a, radius)
    mem_b = _region_indices(points, idx_b, radius)

    # Use the region centroid as the representative endpoint location.
    pt_a = points[mem_a].mean(axis=0) if len(mem_a) else points[idx_a]
    pt_b = points[mem_b].mean(axis=0) if len(mem_b) else points[idx_b]

    ep_a = Endpoint(point=pt_a, index=idx_a, member_indices=mem_a)
    ep_b = Endpoint(point=pt_b, index=idx_b, member_indices=mem_b)
    return ep_a, ep_b, graph
