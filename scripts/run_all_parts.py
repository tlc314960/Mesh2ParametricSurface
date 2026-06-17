"""Run the generalized-cylinder proxy fitter over all wrapped parts.

The spout-like part is the primary, well-validated target. The other wrapped
parts (body / lid / handle / knob) are fitted best-effort with the same
generalized-cylinder model for comparison and may be less accurate.

Output layout:

    output/
        <part_slug>/                  # one directory per fitted part
            proxy.ply
            centerline.ply
            cross_sections.ply
            sampled_points.ply
            params.json
            debug.png
        summary.json                  # aggregated metrics + status per part

Usage:
    python scripts/run_all_parts.py --input input --output output
    python scripts/run_all_parts.py --input input --output output --spout-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.fit_spout_proxy import run as run_fit, part_name_from_mesh, part_slug  # noqa: E402


def discover_parts(input_dir: str):
    wrap_dir = os.path.join(input_dir, "alpha_wrapping_per_part")
    if not os.path.isdir(wrap_dir):
        wrap_dir = input_dir
    parts = []
    for f in sorted(os.listdir(wrap_dir)):
        if f.lower().endswith(".obj"):
            parts.append(os.path.join(wrap_dir, f))
    return parts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="input")
    p.add_argument("--output", default="output")
    p.add_argument("--spout-only", action="store_true",
                   help="Only fit the spout-like part")
    p.add_argument("--proxy-type", choices=["auto", "generalized_cylinder", "revolve"],
                   default="auto",
                   help="Proxy representation; 'auto' picks per part by name")
    p.add_argument("--cross-section", choices=["circle", "ellipse"], default="ellipse")
    p.add_argument("--n-points", type=int, default=20000)
    p.add_argument("--n-sections", type=int, default=40)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class _NS:
    """Lightweight namespace mirroring fit_spout_proxy CLI defaults."""
    def __init__(self, **kw):
        self.input = "input"
        self.output = "output"
        self.mesh = None
        self.part_name = None
        self.out_subdir = None
        self.prefix = ""
        self.n_points = 20000
        self.n_sections = 40
        self.n_iters = 4
        self.cross_section = "ellipse"
        self.slab_width_frac = 0.04
        self.end_skip_frac = 0.08
        self.cap_normal_thresh = 0.7
        self.min_reliability = 0.15
        self.angular_resolution = 48
        self.spline_smooth = None
        self.seed = 0
        self.proxy_type = "auto"
        self.__dict__.update(kw)


def _json_safe(v):
    try:
        import numpy as np
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    return str(v)


def main():
    args = parse_args()
    parts = discover_parts(args.input)
    if not parts:
        print(f"No .obj parts found under {args.input}")
        return

    if args.spout_only:
        parts = [p for p in parts if "spout" in p.lower()]

    os.makedirs(args.output, exist_ok=True)
    summary = {"input": args.input, "output": args.output, "parts": []}

    for mesh_path in parts:
        name = part_name_from_mesh(mesh_path)
        is_spout = "spout" in name.lower()
        slug = part_slug(name)
        print("\n" + "=" * 70)
        print(f"Fitting part: {name} (spout={is_spout})")
        print("=" * 70)
        ns = _NS(
            input=args.input,
            output=args.output,
            mesh=mesh_path,
            part_name=name,
            out_subdir=slug,
            cross_section=args.cross_section,
            n_points=args.n_points,
            n_sections=args.n_sections,
            n_iters=args.n_iters,
            seed=args.seed,
            proxy_type=args.proxy_type,
        )
        entry = {
            "part_name": name,
            "slug": slug,
            "source_mesh_path": mesh_path,
            "is_spout": is_spout,
        }
        try:
            res = run_fit(ns)
            entry["status"] = "ok"
            entry["proxy_type"] = res.get("proxy_type")
            entry["out_dir"] = os.path.relpath(res["out_dir"], args.output)
            entry["metrics"] = res["metrics"]
        except Exception as e:  # best-effort for non-spout parts
            print(f"[run_all_parts] FAILED on {name}: {e}")
            traceback.print_exc()
            entry["status"] = "failed"
            entry["error"] = str(e)
        summary["parts"].append(entry)

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    print("\n" + "=" * 70)
    print(f"Wrote summary: {summary_path}")
    for e in summary["parts"]:
        status = e["status"]
        cov = e.get("metrics", {}).get("coverage_ratio")
        cov_s = f"coverage={cov:.3f}" if isinstance(cov, (int, float)) else ""
        ptype = e.get("proxy_type", "")
        print(f"  {e['slug']:18s} {status:8s} {ptype:22s} {cov_s}")


if __name__ == "__main__":
    main()
