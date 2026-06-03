# Raw Capture to 3DGS Workflow

This workflow prepares independent benchmark scenes from raw photos or videos
without adding a 3D Gaussian Splatting trainer to PlaySplat. PlaySplat remains
responsible for capture organization, frame extraction, image quality checks,
training instructions, staging, and benchmark bookkeeping.

## Folder Convention

Use one folder per scene:

```text
raw_captures/<scene_id>/
  images/              # extracted or copied images
  video/               # optional source video
  notes.md             # capture notes and readiness checklist
  training_commands.md # generated external training instructions
  training_output/     # optional external 3DGS output reference
```

The final external 3DGS output is expected to be:

```text
point_cloud/iteration_xxxxx/point_cloud.ply
```

After training, stage the selected point cloud into PlaySplat:

```text
data/scenes/<scene_id>/point_cloud.ply
```

## Workflow

1. Create a capture checklist.

   ```bash
   python scripts/create_scene_checklist.py --scene-id room01 --category indoor_room --output raw_captures/room01/notes.md
   ```

2. Capture 80-200 images, or record a smooth video and extract frames.

   ```bash
   python scripts/extract_video_frames.py --video "D:/captures/room01/video.mp4" --output-dir raw_captures/room01/images --fps 2
   ```

3. Validate the image set before training.

   ```bash
   python scripts/validate_capture_images.py --images raw_captures/room01/images --output raw_captures/room01/capture_quality_report.json
   ```

4. Generate a reproducible external training plan.

   ```bash
   python scripts/generate_3dgs_training_plan.py --scene-id room01 --images raw_captures/room01/images --output-dir raw_captures/room01 --trainer-name generic
   ```

5. Train externally with the selected 3DGS tool.

   PlaySplat does not run the trainer. Keep the generated commands and any
   edits you made to them in `training_commands.md` for reproducibility.

6. Locate the final trained point cloud.

   ```bash
   python scripts/find_3dgs_point_cloud.py --root "D:/captures/room01/training_output"
   ```

7. Stage the final `.ply` into PlaySplat.

   ```bash
   python scripts/stage_independent_scenes.py --scene scene_id=room01,input="D:/captures/room01/training_output/point_cloud/iteration_30000/point_cloud.ply",category=indoor_room,notes="Independent indoor room scene" --registry configs/scenes.local.yaml --data-root data/scenes --copy
   ```

8. Validate the registry and run the benchmark when at least 5 independent
   `split: benchmark` scenes are ready.

   ```bash
   python scripts/validate_scene_registry.py --scene-registry configs/scenes.local.yaml
   ```

## Capture Guidance

- Target 80-200 images per scene.
- Use stable lighting and avoid rapid exposure changes.
- Cover the whole scene from multiple angles.
- Keep the floor or another walkable surface visible when possible.
- Avoid excessive motion blur.
- Avoid large reflective or transparent surfaces when possible.
- Remove or avoid private, sensitive, or identifying objects.

Good first scenes are `room01`, `corridor01`, `desk01`, `outdoor01`, and
`stairs01`.
