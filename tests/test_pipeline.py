from __future__ import annotations

from playsplat.pipeline import run_pipeline
from playsplat.utils.config import PipelineSettings


def test_pipeline_returns_layered_scene(tmp_path) -> None:
    settings = PipelineSettings(
        scene_id="smoke",
        input_path=None,
        output_dir=tmp_path,
        export_targets=("webgl",),
        semantic_vocabulary=("floor", "wall"),
        affordance_labels=("walkable",),
    )

    result = run_pipeline(settings)

    assert result.scene.metadata.scene_id == "smoke"
    assert result.scene.visual.gaussian_count == 0
    assert result.scene.semantics.labels == ("floor", "wall")
    assert result.scene.affordances.labels == ("walkable",)
    assert result.exports[0].target == "webgl"
    assert result.report.status == "placeholder"
