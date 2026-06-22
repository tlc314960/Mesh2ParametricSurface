"""Export clean, aligned per-part PLY meshes from the original AI FBX.

The FBX is evaluated by Blender so object hierarchy, modifiers, transforms,
coordinate conversion, triangulation, and normals are handled correctly.
Conservative mesh cleanup and alpha-wrap alignment diagnostics then run in the
project Python environment.

Example:

    .venv/bin/python scripts/preprocess_original_fbx_parts.py \
        --fbx input/TeportParts.fbx \
        --alpha-dir input/alpha_wrapping_per_part \
        --output input/original_parts_ply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BLENDER_CANDIDATES = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
)


def _script_argv() -> List[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", default="input/TeportParts.fbx")
    parser.add_argument("--alpha-dir", default="input/alpha_wrapping_per_part")
    parser.add_argument("--output", default="input/original_parts_ply")
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable; auto-detected on macOS/PATH by default",
    )
    parser.add_argument(
        "--min-component-face-ratio",
        type=float,
        default=0.0,
        help=(
            "Remove disconnected components smaller than this fraction of the "
            "largest component. Default 0 preserves every component."
        ),
    )
    parser.add_argument(
        "--no-debug-figure",
        action="store_true",
        help="Skip debug_alignment.png",
    )
    parser.add_argument("--blender-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-manifest", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def find_blender(explicit: str | None) -> str:
    candidates = (explicit,) if explicit else DEFAULT_BLENDER_CANDIDATES
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
        located = shutil.which(expanded)
        if located:
            return located
    raise FileNotFoundError(
        "Blender was not found. Install Blender or pass --blender /path/to/blender."
    )


def _matrix_rows(matrix) -> List[List[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def blender_worker(args) -> None:
    """Run inside Blender and export one transform-baked PLY per mesh object."""
    import bpy
    import bmesh
    from mathutils import Vector

    fbx_path = Path(args.fbx).resolve()
    worker_output = Path(args.worker_output).resolve()
    manifest_path = Path(args.worker_manifest).resolve()
    worker_output.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_objects = sorted(
        (obj for obj in scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name.lower(),
    )
    non_mesh_objects = [
        {"name": obj.name, "type": obj.type}
        for obj in scene.objects
        if obj.type != "MESH"
    ]

    records: List[Dict[str, object]] = []
    for object_index, obj in enumerate(mesh_objects):
        evaluated = obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        baked_mesh.transform(obj.matrix_world)

        bm = bmesh.new()
        bm.from_mesh(baked_mesh)
        non_triangles_before = sum(1 for face in bm.faces if len(face.verts) != 3)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(baked_mesh)
        bm.free()
        baked_mesh.update()

        temporary = bpy.data.objects.new(f"__export_{obj.name}", baked_mesh)
        scene.collection.objects.link(temporary)
        temporary.matrix_world.identity()

        bpy.ops.object.select_all(action="DESELECT")
        temporary.select_set(True)
        bpy.context.view_layer.objects.active = temporary

        raw_name = f"{object_index:02d}_{obj.name}.ply"
        raw_path = worker_output / raw_name
        bpy.ops.wm.ply_export(
            filepath=str(raw_path),
            check_existing=False,
            forward_axis="Y",
            up_axis="Z",
            global_scale=1.0,
            apply_modifiers=False,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_attributes=True,
            export_triangulated_mesh=True,
            ascii_format=False,
        )

        world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        matrix = obj.matrix_world
        determinant = float(matrix.to_3x3().determinant())
        records.append(
            {
                "object_index": object_index,
                "object_name": obj.name,
                "mesh_data_name": obj.data.name,
                "parent_name": obj.parent.name if obj.parent else None,
                "material_names": [material.name for material in obj.data.materials],
                "source_vertex_count": int(len(obj.data.vertices)),
                "source_polygon_count": int(len(obj.data.polygons)),
                "evaluated_vertex_count": int(len(baked_mesh.vertices)),
                "evaluated_triangle_count": int(len(baked_mesh.polygons)),
                "non_triangular_faces_before_export": int(non_triangles_before),
                "matrix_world": _matrix_rows(matrix),
                "matrix_world_determinant": determinant,
                "world_bounds": {
                    "min": [
                        min(float(corner[axis]) for corner in world_corners)
                        for axis in range(3)
                    ],
                    "max": [
                        max(float(corner[axis]) for corner in world_corners)
                        for axis in range(3)
                    ],
                },
                "raw_ply_path": str(raw_path),
            }
        )

        bpy.data.objects.remove(temporary, do_unlink=True)
        bpy.data.meshes.remove(baked_mesh)

    manifest = {
        "fbx_path": str(fbx_path),
        "scene_unit_system": scene.unit_settings.system,
        "scene_unit_scale_length": float(scene.unit_settings.scale_length),
        "scene_object_count": int(len(scene.objects)),
        "mesh_object_count": int(len(mesh_objects)),
        "ignored_non_mesh_object_count": int(len(non_mesh_objects)),
        "ignored_non_mesh_objects": non_mesh_objects,
        "objects": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[original_parts] Blender exported {len(records)} mesh objects")


def _load_worker_mesh(path: str):
    import trimesh

    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected triangle mesh from Blender PLY, got {type(loaded)}")
    return loaded


def run_preprocess(args) -> Dict[str, object]:
    from src import io_utils
    from src.original_part_preprocess import (
        alignment_diagnostics,
        clean_original_mesh,
        discover_alpha_parts,
        display_name,
        match_alpha_part,
        mesh_geometry_summary,
        save_alignment_figure,
        slugify,
        topology_summary,
    )

    fbx_path = _resolve_repo_path(args.fbx)
    alpha_dir = _resolve_repo_path(args.alpha_dir)
    output_dir = _resolve_repo_path(args.output)
    if not fbx_path.is_file():
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")
    if not alpha_dir.is_dir():
        raise FileNotFoundError(f"Alpha-wrap directory not found: {alpha_dir}")
    if args.min_component_face_ratio < 0:
        raise ValueError("--min-component-face-ratio must be non-negative")

    blender = find_blender(args.blender)
    alpha_parts = discover_alpha_parts(alpha_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="m2ps_fbx_parts_") as temp:
        temp_dir = Path(temp)
        worker_output = temp_dir / "raw"
        worker_manifest = temp_dir / "blender_manifest.json"
        command = [
            blender,
            "-b",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--blender-worker",
            "--fbx",
            str(fbx_path),
            "--worker-output",
            str(worker_output),
            "--worker-manifest",
            str(worker_manifest),
        ]
        print(f"[original_parts] Loading FBX with Blender: {fbx_path}")
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        manifest = json.loads(worker_manifest.read_text(encoding="utf-8"))

        mapping_entries = []
        matched_for_figure = []
        warnings: List[str] = []
        used_canonical_names: Dict[str, int] = {}

        for record in manifest["objects"]:
            raw_mesh = _load_worker_mesh(record["raw_ply_path"])
            cleaned, clean_report = clean_original_mesh(
                raw_mesh,
                min_component_face_ratio=args.min_component_face_ratio,
            )
            alpha_part, match_method = match_alpha_part(
                record["object_name"],
                record.get("material_names", []),
                alpha_parts,
            )

            if alpha_part is not None:
                canonical = alpha_part.canonical_name
                part_index = alpha_part.index
                semantic_label = alpha_part.display_name
            else:
                canonical = slugify(record["object_name"]).lower()
                part_index = record["object_index"]
                semantic_label = slugify(record["object_name"])
                warnings.append(f"Unmatched FBX object: {record['object_name']}")

            duplicate_number = used_canonical_names.get(canonical, 0)
            used_canonical_names[canonical] = duplicate_number + 1
            suffix = f"_component_{duplicate_number + 1}" if duplicate_number else ""
            index_text = f"{part_index}_" if part_index is not None else ""
            exported_name = f"part_{index_text}{semantic_label}{suffix}_original.ply"
            exported_path = output_dir / exported_name
            io_utils.save_mesh(str(exported_path), cleaned)

            alignment = None
            if alpha_part is not None:
                alignment = alignment_diagnostics(cleaned, alpha_part.mesh)
                matched_for_figure.append((canonical, cleaned, alpha_part.mesh))
                for warning in alignment["warnings"]:
                    warnings.append(f"{record['object_name']}: {warning}")

            if record["matrix_world_determinant"] < 0:
                warnings.append(
                    f"{record['object_name']}: source object transform has negative determinant"
                )
            topology = topology_summary(cleaned)
            if not topology["winding_consistent"]:
                warnings.append(f"{record['object_name']}: face winding remains inconsistent")
            if topology["nonmanifold_edge_count"]:
                warnings.append(
                    f"{record['object_name']}: "
                    f"{topology['nonmanifold_edge_count']} nonmanifold edges"
                )

            entry = {
                "original_fbx_object_name": record["object_name"],
                "mesh_data_name": record["mesh_data_name"],
                "parent_name": record["parent_name"],
                "material_names": record["material_names"],
                "canonical_part_name": canonical,
                "match_method": match_method,
                "exported_ply_path": os.path.relpath(exported_path, REPO_ROOT),
                "matched_alpha_wrapped_mesh_path": (
                    os.path.relpath(alpha_part.path, REPO_ROOT) if alpha_part else None
                ),
                "source_fbx": {
                    "vertex_count": record["source_vertex_count"],
                    "polygon_count": record["source_polygon_count"],
                    "evaluated_vertex_count": record["evaluated_vertex_count"],
                    "evaluated_triangle_count": record["evaluated_triangle_count"],
                    "non_triangular_faces_before_export": record[
                        "non_triangular_faces_before_export"
                    ],
                    "matrix_world": record["matrix_world"],
                    "matrix_world_determinant": record["matrix_world_determinant"],
                    "world_bounds": record["world_bounds"],
                },
                "exported_mesh": mesh_geometry_summary(cleaned),
                "topology_and_normals": topology,
                "cleanup": clean_report,
                "alignment": alignment,
            }
            mapping_entries.append(entry)

        mapping_payload = {
            "description": (
                "Mapping from original AI-generated FBX mesh objects to canonical "
                "pipeline part names. Edit canonical_part_name only if an automatic "
                "match is incorrect, then rerun with appropriately named source data."
            ),
            "fbx_path": os.path.relpath(fbx_path, REPO_ROOT),
            "alpha_wrap_directory": os.path.relpath(alpha_dir, REPO_ROOT),
            "parts": mapping_entries,
        }
        mapping_path = output_dir / "part_mapping.json"
        mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

        debug_path = output_dir / "debug_alignment.png"
        if not args.no_debug_figure:
            save_alignment_figure(debug_path, matched_for_figure)

        summary = {
            "fbx_path": os.path.relpath(fbx_path, REPO_ROOT),
            "alpha_wrap_directory": os.path.relpath(alpha_dir, REPO_ROOT),
            "output_directory": os.path.relpath(output_dir, REPO_ROOT),
            "blender_executable": blender,
            "scene_unit_system": manifest["scene_unit_system"],
            "scene_unit_scale_length": manifest["scene_unit_scale_length"],
            "scene_object_count": manifest["scene_object_count"],
            "mesh_object_count": manifest["mesh_object_count"],
            "ignored_non_mesh_object_count": manifest["ignored_non_mesh_object_count"],
            "ignored_non_mesh_objects": manifest["ignored_non_mesh_objects"],
            "parts_exported": len(mapping_entries),
            "exported_files": [entry["exported_ply_path"] for entry in mapping_entries],
            "all_matched_parts_aligned": bool(
                mapping_entries
                and all(
                    entry["alignment"]
                    and entry["alignment"]["coordinate_systems_appear_aligned"]
                    for entry in mapping_entries
                )
            ),
            "global_transform_correction": None,
            "global_transform_note": (
                "No correction applied. Blender FBX world coordinates already align "
                "with the alpha-wrapped meshes; object transforms were baked per object."
            ),
            "component_filter_face_ratio": args.min_component_face_ratio,
            "warnings": sorted(set(warnings)),
            "part_mapping_path": os.path.relpath(mapping_path, REPO_ROOT),
            "debug_alignment_path": (
                os.path.relpath(debug_path, REPO_ROOT)
                if debug_path.exists()
                else None
            ),
        }
        summary_path = output_dir / "preprocess_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[original_parts] Export mapping")
    print(
        f"{'FBX object':16s} {'canonical':16s} {'vertices':>10s} "
        f"{'faces':>10s}  exported file"
    )
    for entry in mapping_entries:
        geometry = entry["exported_mesh"]
        print(
            f"{entry['original_fbx_object_name']:16s} "
            f"{entry['canonical_part_name']:16s} "
            f"{geometry['vertex_count']:10d} {geometry['face_count']:10d}  "
            f"{entry['exported_ply_path']}"
        )
        alignment = entry["alignment"]
        if alignment:
            original_bounds = alignment["original"]["bounds"]
            alpha_bounds = alignment["alpha_wrapped"]["bounds"]
            print(
                " " * 4
                + f"scale_ratio={alignment['approximate_scale_ratio']:.6f}, "
                + f"centroid_delta={alignment['centroid_difference_norm']:.6g}, "
                + f"aligned={alignment['coordinate_systems_appear_aligned']}"
            )
            print(
                " " * 4
                + f"original bounds={original_bounds['min']} -> {original_bounds['max']}"
            )
            print(
                " " * 4
                + f"alpha bounds   ={alpha_bounds['min']} -> {alpha_bounds['max']}"
            )

    print(f"[original_parts] Mapping: {mapping_path}")
    print(f"[original_parts] Summary: {summary_path}")
    if debug_path.exists():
        print(f"[original_parts] Debug figure: {debug_path}")
    if summary["warnings"]:
        print("[original_parts] Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    return summary


def main():
    args = parse_args(_script_argv())
    if args.blender_worker:
        if not args.worker_output or not args.worker_manifest:
            raise ValueError("Blender worker requires --worker-output and --worker-manifest")
        blender_worker(args)
        return
    run_preprocess(args)


if __name__ == "__main__":
    main()
