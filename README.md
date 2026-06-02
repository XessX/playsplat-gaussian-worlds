# PlaySplat

PlaySplat is a research prototype for **Playability-Aware Gaussian Splatting for Physics-Ready Interactive 3D Worlds**.

The long-term goal is to convert 3D Gaussian Splatting scenes into layered interactive scene representations that can be simulated, navigated, annotated, and exported to real-time engines.

## Research Goal

3D Gaussian Splatting produces high-quality visual reconstructions, but those reconstructions are not directly playable. They usually lack explicit surfaces, stable collision geometry, traversability labels, semantic structure, and affordances. PlaySplat explores how to bridge that gap.

The target representation contains:

1. A visual Gaussian splat layer.
2. A proxy geometry layer.
3. A collision and physics layer.
4. A navigation and walkable layer.
5. A semantic scene layer.
6. An affordance layer.
7. Export targets for Unity, PlayCanvas, and WebGL.

This first version is intentionally lightweight. It contains a clean Python package structure, typed placeholder APIs, a default config, and a CLI skeleton for running the future pipeline.

## Planned Pipeline

```text
Gaussian scene input
        |
        v
Visual splat layer
        |
        v
Proxy geometry extraction
        |
        v
Collision and physics approximation
        |
        v
Navigation and walkability estimation
        |
        v
Semantic scene understanding
        |
        v
Affordance inference
        |
        v
Unity / PlayCanvas / WebGL export
```

## Repository Layout

```text
playsplat/
  configs/              YAML experiment and pipeline configs
  data/                 Local datasets and scene inputs
  docs/                 Notes, papers, diagrams, and design docs
  outputs/              Generated artifacts and exports
  scripts/              Research scripts and CLI entry points
  src/playsplat/        Python package source
    io/                 Scene loading and serialization
    gaussian/           Gaussian splat data structures and visual layer logic
    geometry/           Proxy meshes and geometry extraction
    physics/            Collision and physics-ready approximations
    navigation/         Walkability, navmesh, and agent traversal
    semantics/          Semantic scene representations
    affordance/         Interaction affordance inference
    export/             Engine and WebGL export interfaces
    evaluation/         Metrics and benchmark helpers
    utils/              Config, logging, and shared utilities
  tests/                Unit and smoke tests
```

## Quick Start

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/run_pipeline.py --config configs/default.yaml
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python .\scripts\run_pipeline.py --config .\configs\default.yaml
```

## Current Status

The current implementation is a scaffold. The functions return placeholder typed objects so researchers can start wiring experiments, tests, and notebooks without committing to final algorithms too early.

## Engine Export Bundles

PlaySplat can generate research-ready export bundles for Unity, PlayCanvas, and generic WebGL. Bundles are written under `outputs/exports/<target>/` and contain a `manifest.json`, target-specific import README, available visual splat references, proxy collision geometry, scene-structure meshes, and metadata files.

These bundles are packaging artifacts, not full engine integrations. They document how to import the visual Gaussian splat, use `collision_mesh.obj` for physics, keep proxy meshes for inspection, use floor or walkable meshes for navigation, and treat wall or obstacle meshes as blockers.

## Collision Mesh Simplification

PlaySplat now writes a simplified `collision_mesh.obj` alongside the higher-detail `proxy_mesh.obj` when geometry simplification is enabled. The collision mesh is generated with deterministic vertex clustering, records before/after face counts in `proxy_metadata.json`, and is intended as the first candidate for MeshCollider or physics-engine collision import.

The original proxy mesh is preserved for structure detection, debugging, and higher-detail inspection. Floor, wall, obstacle, and walkable meshes remain labeled region outputs for navigation and visual debugging rather than the primary whole-scene collider.

## Geometry-Derived Semantics and Affordances

PlaySplat currently uses deterministic geometry-derived labels when scene structure is available. Detected floor, wall, obstacle, and walkable regions are converted into baseline semantic labels such as `floor`, `wall`, `obstacle`, and `walkable_surface`.

Those labels then produce simple affordances: floor and walkable surfaces become `walkable` and `support`, walls become `blocking`, and obstacles become `blocking` plus `interactable_candidate`. These are baseline labels for research bookkeeping, not learned semantic segmentation.

When geometry structure is missing, semantic and affordance layers are marked as placeholders. Placeholder layers are reported in metrics and manifests, but they do not count as fully ready in the playability score.

## Playability Metrics

PlaySplat writes `playability_report.json` and `playability_metrics.csv` for each CLI run. The current metrics measure prototype readiness and scene-layer completeness: visual Gaussian availability, proxy mesh complexity, walkable area, obstacle and wall areas, navigation readiness, collision readiness, and export readiness.

The overall score is an early deterministic research signal, not a final benchmark score. It is intended to make pipeline progress measurable while the representation, algorithms, and evaluation protocol continue to evolve.

## Running Real Scene Experiments

Use `scripts/run_scene_experiment.py` to run PlaySplat over one or more real Gaussian Splat `.ply` scenes and collect summary outputs.

```bash
python scripts/run_scene_experiment.py --input path/to/scene.ply --scene-id scene_name --output-root outputs/experiments --config configs/default.yaml
```

For a batch:

```bash
python scripts/run_scene_experiment.py --input scene1.ply scene2.ply scene3.ply --output-root outputs/experiments --config configs/default.yaml
```

Each scene writes normal pipeline artifacts under `outputs/experiments/<scene_id>/`. The runner also writes `experiment_summary.csv` and `experiment_summary.json` with scalar playability metrics and failure records for scenes that could not be processed.

## Generating Debug Previews

Use `scripts/generate_previews.py` to create quick PNG previews for a processed scene:

```bash
python scripts/generate_previews.py --scene-output outputs/experiments/scene1
```

To generate previews during an experiment run:

```bash
python scripts/run_scene_experiment.py --input data/scenes/scene1/point_cloud.ply --scene-id scene1 --generate-previews
```

Previews are written under `outputs/experiments/<scene_id>/previews/` when the corresponding files exist. Current preview targets include Gaussian point distribution, proxy mesh, collision mesh, floor/wall/obstacle/walkable meshes, and the playability summary card.

## Planned Milestones

1. Define Gaussian scene input formats and metadata conventions.
2. Build proxy geometry extraction baselines from splat density and depth cues.
3. Generate collision primitives and static rigid-body approximations.
4. Estimate walkable surfaces and navigation regions.
5. Attach open-vocabulary semantic labels to scene regions and objects.
6. Infer affordances such as walkable, climbable, sit-able, movable, openable, and interactable.
7. Add export adapters for Unity, PlayCanvas, and WebGL.
8. Evaluate playability with collision validity, navigation success, interaction coverage, and visual alignment metrics.

## Development

Run tests with:

```bash
pytest
```

The code is designed to stay research-friendly: small modules, typed dataclasses, simple interfaces, and room for experimental methods to be swapped in as the project evolves.
