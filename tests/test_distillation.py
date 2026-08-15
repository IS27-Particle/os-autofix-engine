"""Unit tests for Edge Model Distillation Pipeline and Quantization."""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer.distillation_pipeline import (
    DistillationConfig,
    DistillationPipeline,
    kl_divergence,
    softmax,
)


def test_compute_kd_loss_components() -> None:
    """Test soft Cross-Entropy and KL divergence loss calculations."""
    cfg = DistillationConfig(temperature=2.0, alpha_kd=0.5)
    pipeline = DistillationPipeline(cfg)

    # Identical logits -> zero KL divergence
    logits_t = [2.0, 1.0, 0.0]
    logits_s = [2.0, 1.0, 0.0]

    p = softmax(logits_t, 2.0)
    q = softmax(logits_s, 2.0)
    assert round(kl_divergence(p, q), 4) == 0.0

    total_loss, ce_loss, kl_loss = pipeline.compute_kd_loss(
        student_logits=logits_s,
        teacher_logits=logits_t,
        target_idx=0,
    )

    assert total_loss > 0.0
    assert ce_loss > 0.0
    assert kl_loss == 0.0


def test_onnx_and_gguf_export_structure(tmp_path: Path) -> None:
    """Verify generated ONNX and GGUF manifest structures."""
    cfg = DistillationConfig(quantization="q4_k_m")
    pipeline = DistillationPipeline(cfg)

    onnx_path = pipeline.export_onnx(tmp_path, "test_edge")
    assert Path(onnx_path).exists()
    assert "ir_version" in Path(onnx_path).read_text(encoding="utf-8")

    gguf_path = pipeline.export_gguf(tmp_path, "test_edge", quant_type="q4_k_m")
    assert Path(gguf_path).exists()
    assert "GGUF" in Path(gguf_path).read_text(encoding="utf-8")

    modelfile_path = pipeline.generate_edge_modelfile(tmp_path, gguf_path)
    assert Path(modelfile_path).exists()
    assert "FROM" in Path(modelfile_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_distillation_pipeline_execution(tmp_path: Path) -> None:
    """Test full async distillation execution workflow."""
    cfg = DistillationConfig(
        teacher_model="qwen2.5-coder:7b",
        student_model="qwen2.5-coder:0.5b",
        export_onnx=True,
        export_gguf=True,
    )
    pipeline = DistillationPipeline(cfg)

    result = await pipeline.run_distillation(
        dataset_path="dummy.jsonl",
        output_dir=str(tmp_path),
    )

    assert result.distillation_id.startswith("distill-")
    assert result.teacher_model == "qwen2.5-coder:7b"
    assert result.student_model == "qwen2.5-coder:0.5b"
    assert result.final_loss > 0.0
    assert result.onnx_path is not None
    assert result.gguf_path is not None
    assert result.model_size_mb > 0.0
