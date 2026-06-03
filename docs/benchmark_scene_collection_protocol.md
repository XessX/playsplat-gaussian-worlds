# Benchmark Scene Collection Protocol

This protocol defines how to collect independent benchmark scenes for PlaySplat:
Playability-Aware Gaussian Splatting for Physics-Ready Interactive 3D Worlds.

## Purpose

Independent benchmark scenes provide evidence that PlaySplat can convert real
3D Gaussian Splatting reconstructions into layered interactive scene
representations beyond the internal debug case. Each benchmark scene should be a
separately captured or independently sourced 3DGS reconstruction with a clear
provenance trail from capture to trained point cloud to staged PlaySplat input.

The minimum target is 5 independent benchmark scenes. A stronger target is 8
independent benchmark scenes spanning diverse geometry and capture conditions.

## Evidence Boundaries

Do not count Scientific Reports scenes, PlaySplat generated outputs, or internal
debug scenes as independent benchmark evidence.

Scientific Reports scenes and sparse-view project artifacts may be useful for
development context, but they are not independent PlaySplat benchmark scenes
unless their capture, license, and experimental role are explicitly cleared.

PlaySplat outputs are generated artifacts, not fresh inputs. Counting generated
proxy meshes, exports, previews, or output folders as benchmark inputs would mix
evaluation data with results.

Debug scenes are useful for smoke tests, visualization checks, and local
development. They should remain marked as `source: internal_debug` and
`split: debug` so they do not contaminate paper evidence.

## Recommended Categories

Collect scenes across these categories where possible:

- `indoor_room`
- `corridor`
- `furniture_heavy`
- `outdoor_open`
- `outdoor_complex`
- `stairs`
- `object_cluster`
- `sparse_structure`

## Capture Checklist

For each scene:

- Collect 80-200 images, or record a smooth video and convert it to frames.
- Cover the full scene from multiple angles, with enough overlap for 3DGS
  training.
- Include the floor or another walkable surface when possible.
- Avoid excessive motion blur.
- Avoid large reflective or transparent surfaces when possible.
- Keep lighting reasonably stable throughout the capture.
- Avoid private or sensitive objects, people, screens, documents, and personal
  identifiers.

## Expected 3DGS Output

After 3DGS training, the expected point cloud path is usually:

```text
point_cloud/iteration_xxxxx/point_cloud.ply
```

Record the actual training iteration in the intake sheet.

## Expected PlaySplat Staging Path

Stage the final input for PlaySplat at:

```text
data/scenes/<scene_id>/point_cloud.ply
```

The staged file should be referenced in the local registry as:

```yaml
input_path: data/scenes/<scene_id>/point_cloud.ply
source: independent
split: benchmark
```

Keep private raw capture paths and local machine paths in ignored local files
such as `configs/scenes.local.yaml` or the intake sheet used during collection.
