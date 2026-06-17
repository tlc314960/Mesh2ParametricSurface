"""Fit a generalized-cylinder proxy to a spout-like alpha-wrapped part.

Usage (defaults adapted to the actual repo input layout):

    python scripts/fit_spout_proxy.py \
        --input input \
        --output output

By default it picks the alpha-wrapped spout part
(input/alpha_wrapping_per_part/part_4_Spout_wrapped.obj). Override with
--mesh to fit any specific wrapped part.

Output layout (per part):

    output/<part_slug>/
        proxy.ply
        centerline.ply
        cross_sections.ply
        sampled_points.ply
        params.json
        debug.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

# Allow running as a script: add repo root to sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src import io_utils, mesh_preprocess, sampling, evaluation, proxy_export, visualization
from src.swept_tube_fitter import FitConfig, fit_swept_tube
from src.revolve_surface import RevolveConfig, fit_revolve_surface


DEFAULT_SPOUT_REL = os.path.join(
    "alpha_wrapping_per_part", "part_4_Spout_wrapped.obj"
)

# Parts whose geometry is a surface of revolution (round container / disc /
# dome) rather than a tube swept along a curved centerline.
REVOLVE_KEYWORDS = ("body", "lid", "knob")
TUBE_KEYWORDS = ("spout", "handle")


def auto_proxy_type(part_name: str) -> str:
    """Choose a proxy type from the part name.

    Tube-like parts (spout, handle) -> generalized_cylinder.
    Round / revolution bodies (body, lid, knob) -> revolve.
    Defaults to generalized_cylinder when unknown.
    """
    low = part_name.lower()
    for kw in REVOLVE_KEYWORDS:
        if kw in low:
            return "revolve"
    for kw in TUBE_KEYWORDS:
        if kw in low:
            return "generalized_cylinder"
    return "generalized_cylinder"


def part_slug(name: str) -> str:
    """Filesystem-safe slug for a part name (used as its output sub-dir)."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    return slug or "part"


def part_name_from_mesh(path: str) -> str:
    """Derive a human-readable part name from a wrapped mesh filename."""
    base = os.path.splitext(os.path.basename(path))[0]
    base = base.replace("_wrapped", "")
    # Strip a leading 'part_<n>_' index if present.
    base = re.sub(r"^part_\d+_", "", base)
    return base.replace("_", " ").strip() or base


def find_spout_mesh(input_dir: str) -> str:
    """Locate the spout-like wrapped mesh inside the input directory."""
    candidate = os.path.join(input_dir, DEFAULT_SPOUT_REL)
    if os.path.exists(candidate):
        return candidate
    # Fallback: search for any *Spout*wrapped*.obj
    wrap_dir = os.path.join(input_dir, "alpha_wrapping_per_part")
    search_dir = wrap_dir if os.path.isdir(wrap_dir) else input_dir
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".obj") and "spout" in f.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(
        f"Could not find a spout wrapped .obj under {input_dir}"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="input", help="Input directory")
    p.add_argument("--output", default="output", help="Output directory")
    p.add_argument("--mesh", default=None,
                   help="Explicit path to a wrapped part mesh (overrides --input lookup)")
    p.add_argument("--part-name", default=None,
                   help="Part name; defaults to the one derived from the mesh filename")
    p.add_argument("--out-subdir", default=None,
                   help="Per-part output sub-directory name; defaults to the part slug")
    p.add_argument("--proxy-type", choices=["auto", "generalized_cylinder", "revolve"],
                   default="auto",
                   help="Proxy representation; 'auto' picks by part name")
    p.add_argument("--prefix", default="",
                   help="Optional filename prefix (default: clean names, dir carries identity)")
    p.add_argument("--n-points", type=int, default=20000)
    p.add_argument("--n-sections", type=int, default=40)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--cross-section", choices=["circle", "ellipse"], default="ellipse")
    p.add_argument("--slab-width-frac", type=float, default=0.04)
    p.add_argument("--end-skip-frac", type=float, default=0.08)
    p.add_argument("--cap-normal-thresh", type=float, default=0.7)
    p.add_argument("--min-reliability", type=float, default=0.15)
    p.add_argument("--angular-resolution", type=int, default=48)
    p.add_argument("--spline-smooth", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run(args) -> dict:
    mesh_path = args.mesh or find_spout_mesh(args.input)
    part_name = args.part_name or part_name_from_mesh(mesh_path)
    subdir = args.out_subdir or part_slug(part_name)
    out_dir = os.path.join(args.output, subdir)

    print(f"[fit_proxy] Part: {part_name}  ->  {out_dir}")
    print(f"[fit_proxy] Loading mesh: {mesh_path}")

    mesh = io_utils.load_mesh(mesh_path)
    mesh = mesh_preprocess.clean_mesh(mesh)
    scale = mesh_preprocess.mesh_scale(mesh)
    print(f"[fit_proxy] Cleaned mesh: {len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} faces, scale={scale:.4f}")

    points, normals = sampling.sample_surface(mesh, n_points=args.n_points, seed=args.seed)

    io_utils.ensure_dir(out_dir)
    prefix = args.prefix
    sp_name = f"{prefix}_sampled_points.ply" if prefix else "sampled_points.ply"
    io_utils.save_point_cloud(os.path.join(out_dir, sp_name), points)

    proxy_type = args.proxy_type
    if proxy_type == "auto":
        proxy_type = auto_proxy_type(part_name)
    print(f"[fit_proxy] Proxy type: {proxy_type}")

    if proxy_type == "revolve":
        return _run_revolve(args, mesh_path, part_name, subdir, out_dir,
                            points, normals, scale, prefix)

    cfg = FitConfig(
        n_sections=args.n_sections,
        n_iters=args.n_iters,
        cross_section_kind=args.cross_section,
        slab_width_frac=args.slab_width_frac,
        end_skip_frac=args.end_skip_frac,
        cap_normal_thresh=args.cap_normal_thresh,
        min_reliability=args.min_reliability,
        angular_resolution=args.angular_resolution,
        spline_smooth=args.spline_smooth,
        seed=args.seed,
    )

    print("[fit_proxy] Fitting swept tube proxy ...")
    result = fit_swept_tube(points, normals, cfg)

    metrics = evaluation.evaluate(points, result, scale)
    print("[fit_proxy] Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    paths = proxy_export.export_proxy(
        out_dir=out_dir,
        part_name=part_name,
        source_mesh_path=mesh_path,
        result=result,
        metrics=metrics,
        prefix=prefix,
    )

    dbg_name = f"{prefix}_debug.png" if prefix else "debug.png"
    debug_path = os.path.join(out_dir, dbg_name)
    visualization.save_debug_figure(debug_path, points, result)
    paths["debug"] = debug_path

    print("[fit_proxy] Wrote:")
    for k, v in paths.items():
        print(f"    {k}: {v}")

    return {
        "part_name": part_name,
        "subdir": subdir,
        "out_dir": out_dir,
        "source_mesh_path": mesh_path,
        "proxy_type": "generalized_cylinder",
        "paths": paths,
        "metrics": metrics,
    }


def _run_revolve(args, mesh_path, part_name, subdir, out_dir,
                 points, normals, scale, prefix) -> dict:
    cfg = RevolveConfig(
        n_iters=args.n_iters,
        angular_resolution=args.angular_resolution,
        min_reliability=args.min_reliability,
        profile_smooth=args.spline_smooth,
        seed=args.seed,
    )

    print("[fit_proxy] Fitting surface-of-revolution proxy ...")
    result = fit_revolve_surface(points, normals, cfg)

    metrics = evaluation.evaluate_revolve(points, result, scale)
    print("[fit_proxy] Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    paths = proxy_export.export_revolve(
        out_dir=out_dir,
        part_name=part_name,
        source_mesh_path=mesh_path,
        result=result,
        metrics=metrics,
        prefix=prefix,
    )

    dbg_name = f"{prefix}_debug.png" if prefix else "debug.png"
    debug_path = os.path.join(out_dir, dbg_name)
    visualization.save_revolve_debug_figure(debug_path, points, result)
    paths["debug"] = debug_path

    print("[fit_proxy] Wrote:")
    for k, v in paths.items():
        print(f"    {k}: {v}")

    return {
        "part_name": part_name,
        "subdir": subdir,
        "out_dir": out_dir,
        "source_mesh_path": mesh_path,
        "proxy_type": "surface_of_revolution",
        "paths": paths,
        "metrics": metrics,
    }


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
