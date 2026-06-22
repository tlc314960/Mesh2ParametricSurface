# Original AI FBX part preprocessing

This preprocessing path is separate from proxy fitting:

- `input/alpha_wrapping_per_part/` remains the source for smooth parametric
  proxy fitting.
- `input/TeportParts.fbx` is the source of original visual surface detail.
- `input/original_parts_ply/` contains transform-baked, conservatively cleaned
  original meshes for later detailed-surface sampling.

No alpha wrapping, smoothing, simplification, remeshing, shell construction, or
outer-layer extraction is performed here.

## Run

Blender is required because trimesh does not import FBX. The public command is
still run from the project Python environment; it launches Blender headlessly
for FBX scene evaluation:

```bash
.venv/bin/python scripts/preprocess_original_fbx_parts.py \
    --fbx input/TeportParts.fbx \
    --alpha-dir input/alpha_wrapping_per_part \
    --output input/original_parts_ply
```

On macOS the script auto-detects
`/Applications/Blender.app/Contents/MacOS/Blender`. Elsewhere, install Blender
on `PATH` or pass `--blender /path/to/blender`.

The default cleanup preserves every connected component. To remove components
below an explicit face-count ratio of the largest component:

```bash
--min-component-face-ratio 0.001
```

Use this only after inspecting `part_mapping.json`; disconnected components can
be intentional visual details.

## Current teapot mapping

The FBX contains five generic object names and no materials. Numeric indices
match the alpha-wrapped inputs:

| FBX object | Canonical part | Export |
| --- | --- | --- |
| `part_0` | `lid` | `part_0_Lid_original.ply` |
| `part_1` | `teapot_body` | `part_1_Teapot_Body_original.ply` |
| `part_2` | `knob` | `part_2_Knob_original.ply` |
| `part_3` | `handle` | `part_3_Handle_original.ply` |
| `part_4` | `spout` | `part_4_Spout_original.ply` |

Semantic object/material names are preferred for other FBX files; shared
`part_N` indices are the fallback.

## Processing and diagnostics

Blender:

1. Imports the complete FBX scene.
2. Ignores non-mesh objects.
3. Evaluates modifiers.
4. Bakes each object's world transform into its vertices.
5. Triangulates polygons.
6. Exports normals and geometry to an intermediate PLY.

The project Python environment then:

1. Merges duplicate vertices.
2. Removes degenerate and duplicate faces.
3. Removes unreferenced vertices.
4. Reports connected components without deleting them by default.
5. Repairs inconsistent winding when possible.
6. Inverts only closed meshes with negative signed volume.
7. Computes face and vertex normals.
8. Compares bounds, centroids, scale, and per-axis extents with the matching
   alpha-wrapped mesh.

Outputs:

```text
input/original_parts_ply/
  part_*_original.ply
  part_mapping.json
  preprocess_summary.json
  debug_alignment.png
```

`part_mapping.json` contains object transforms, geometry/topology statistics,
normal diagnostics, matching method, alpha mesh path, and per-part alignment
results. `preprocess_summary.json` records scene-level status and warnings.

For the current teapot, no global transform correction is needed: Blender FBX
world coordinates already match the alpha-wrapped meshes in scale, translation,
axis order, and Z-up orientation. Object transforms are nevertheless baked
explicitly during every run.
