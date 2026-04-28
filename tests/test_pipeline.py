"""Tests for pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen36gguf.config import QuantizationConfig
from qwen36gguf.pipeline import Pipeline


class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(llama_cpp_dir=llama_cpp)
            pipeline = Pipeline(config)

            assert pipeline.config == config
            assert pipeline.results == {}

    def test_dry_run_pipeline(self):
        """Test pipeline in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            # Should complete without errors in dry run mode
            result = pipeline.run()
            assert result is True

    def test_pipeline_validates_binaries(self):
        """Test that pipeline validates binaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            # Should not raise in dry run
            pipeline._validate_binaries()

    def test_pipeline_download_dry_run(self):
        """Test download step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            result = pipeline._download()
            assert result.name == "model"

    def test_pipeline_convert_dry_run(self):
        """Test convert step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            model_path = Path("dummy/model")
            result = pipeline._convert(model_path)
            assert result.name == "model-f16.gguf"

    def test_pipeline_quantize_dry_run(self):
        """Test quantize step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
                quant_levels=["Q4_K_M", "Q8_0"],
            )
            pipeline = Pipeline(config)

            gguf_path = Path("dummy/model-f16.gguf")
            results = pipeline._quantize(gguf_path)

            assert len(results) == 2
            assert any("Q4_K_M" in str(r) for r in results)
            assert any("Q8_0" in str(r) for r in results)

    def test_pipeline_benchmark_dry_run(self):
        """Test benchmark step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            quantized_paths = [Path("model-Q4_K_M.gguf")]
            results = pipeline._benchmark(quantized_paths)

            assert results == {}

    def test_pipeline_generate_card_dry_run(self):
        """Test generate card step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            benchmark_results = {}
            result = pipeline._generate_card(benchmark_results)

            assert result.name == "MODEL_CARD.md"

    def test_pipeline_upload_dry_run(self):
        """Test upload step in dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                dry_run=True,
                hf_repo_id="user/repo",
            )
            pipeline = Pipeline(config)

            benchmark_results = {}
            # Should not raise in dry run
            pipeline._upload(benchmark_results)


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_pipeline_handles_exception(self):
        """Test that pipeline handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                verbose=False,  # Don't re-raise in non-verbose mode
            )
            pipeline = Pipeline(config)

            # Force an error by making validate_binaries fail
            with patch.object(config, "validate_binaries", side_effect=RuntimeError("Binary missing")):
                result = pipeline.run()
                assert result is False

    def test_pipeline_re_raises_in_verbose_mode(self):
        """Test that pipeline re-raises in verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                verbose=True,
            )
            pipeline = Pipeline(config)

            with (
                patch.object(config, "validate_binaries", side_effect=RuntimeError("Binary missing")),
                pytest.raises(RuntimeError, match="Binary missing"),
            ):
                pipeline.run()


class TestPipelineEndToEnd:
    """Pipeline orchestration tests with all subprocess/network calls mocked."""

    def test_full_pipeline_assembles_hf_export(self):
        """All 7 steps run, hf_export bundle is assembled, no upload without repo_id."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            llama_cpp = tmp / "llama.cpp"
            llama_cpp.mkdir()
            output_dir = tmp / "out"

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                model_id="test/model",
                output_dir=output_dir,
                quant_levels=["Q4_K_M"],
                hf_repo_id=None,  # no upload
            )
            pipeline = Pipeline(config)

            # Pre-create paths the mocks will "produce"
            output_dir.mkdir(parents=True, exist_ok=True)
            f16 = output_dir / "model-f16.gguf"
            f16.write_bytes(b"f16 fake")
            quant = output_dir / "model-Q4_K_M.gguf"
            quant.write_bytes(b"q4 fake")
            hf_model_dir = output_dir / "hf_model"
            hf_model_dir.mkdir()

            with (
                patch.object(config, "validate_binaries"),
                patch("qwen36gguf.pipeline.download_model", return_value=hf_model_dir),
                patch("qwen36gguf.pipeline.convert_model", return_value=f16),
                patch("qwen36gguf.pipeline.quantize_model", return_value=quant),
                patch(
                    "qwen36gguf.pipeline.benchmark_model",
                    return_value={
                        "model": "model-Q4_K_M.gguf",
                        "file_size_mb": 10.0,
                        "perplexity": 7.5,
                        "throughput_tok_per_sec": 100.0,
                    },
                ),
                patch("qwen36gguf.pipeline.upload_to_huggingface") as mock_upload,
            ):
                ok = pipeline.run()

            assert ok is True
            bundle = output_dir / "hf_export"
            assert bundle.is_dir()
            assert (bundle / "README.md").is_file()
            assert (bundle / ".gitattributes").is_file()
            assert (bundle / "model-Q4_K_M.gguf").is_file()
            mock_upload.assert_not_called()  # no hf_repo_id → no upload

    def test_pipeline_upload_called_when_repo_set(self):
        """When hf_repo_id is set, upload_to_huggingface receives the bundle path."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            llama_cpp = tmp / "llama.cpp"
            llama_cpp.mkdir()
            output_dir = tmp / "out"

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                model_id="test/model",
                output_dir=output_dir,
                quant_levels=["Q4_K_M"],
                hf_repo_id="user/repo",
            )
            pipeline = Pipeline(config)

            output_dir.mkdir(parents=True, exist_ok=True)
            f16 = output_dir / "model-f16.gguf"
            f16.write_bytes(b"f16")
            quant = output_dir / "model-Q4_K_M.gguf"
            quant.write_bytes(b"q4")

            with (
                patch.object(config, "validate_binaries"),
                patch("qwen36gguf.pipeline.download_model", return_value=output_dir / "hf_model"),
                patch("qwen36gguf.pipeline.convert_model", return_value=f16),
                patch("qwen36gguf.pipeline.quantize_model", return_value=quant),
                patch("qwen36gguf.pipeline.benchmark_model", return_value={"file_size_mb": 1.0}),
                patch("qwen36gguf.pipeline.upload_to_huggingface") as mock_upload,
            ):
                (output_dir / "hf_model").mkdir(exist_ok=True)
                ok = pipeline.run()

            assert ok is True
            mock_upload.assert_called_once()
            kwargs = mock_upload.call_args.kwargs
            assert kwargs["bundle_dir"] == output_dir / "hf_export"
            assert kwargs["repo_id"] == "user/repo"

    def test_dry_run_skips_real_assembly(self):
        """dry_run=True does not write any files in the output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            llama_cpp = tmp / "llama.cpp"
            llama_cpp.mkdir()
            output_dir = tmp / "out"

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                output_dir=output_dir,
                dry_run=True,
            )
            pipeline = Pipeline(config)

            ok = pipeline.run()
            assert ok is True
            # dry-run never creates real artifacts
            assert not (output_dir / "hf_export").exists() or not any(
                (output_dir / "hf_export").iterdir()
            )
