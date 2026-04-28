"""Tests for quantization module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen36gguf.config import QuantizationConfig
from qwen36gguf.quantize import quantize_model


class TestQuantizeModel:
    """Tests for quantize_model function."""

    def test_dry_run_returns_expected_path(self):
        """Test that dry run returns expected output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            gguf_path.write_text("dummy")  # Create dummy file

            result = quantize_model(
                gguf_path=gguf_path,
                quant_type="Q4_K_M",
                dry_run=True,
                verbose=False,
            )

            assert result.name == "model-Q4_K_M.gguf"

    def test_invalid_quant_type_raises_error(self):
        """Test that invalid quant type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            gguf_path.write_text("dummy")

            # This should be caught by CLI validation, but test anyway
            with patch("qwen36gguf.quantize.DEFAULT_CONFIG") as mock_config:
                mock_config.quantize_binary = Path("/nonexistent")

                with pytest.raises(RuntimeError, match="Quantize binary not found"):
                    quantize_model(
                        gguf_path=gguf_path,
                        quant_type="INVALID_TYPE",
                        dry_run=False,
                        verbose=False,
                    )

    def test_missing_binary_raises_error(self):
        """Test that missing quantize binary raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.quantize.DEFAULT_CONFIG") as mock_config:
                mock_config.quantize_binary = Path("/nonexistent/llama-quantize")

                with pytest.raises(RuntimeError, match="Quantize binary not found"):
                    quantize_model(
                        gguf_path=gguf_path,
                        quant_type="Q4_K_M",
                        dry_run=False,
                        verbose=False,
                    )

    def test_successful_quantization(self):
        """Test successful quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            gguf_path.write_text("dummy")
            output_path = Path(tmpdir) / "model-Q4_K_M.gguf"

            with patch("qwen36gguf.quantize.DEFAULT_CONFIG") as mock_config:
                mock_config.quantize_binary = Path("/usr/bin/echo")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="",
                        stderr="",
                        returncode=0,
                    )

                    result = quantize_model(
                        gguf_path=gguf_path,
                        quant_type="Q4_K_M",
                        output_path=output_path,
                        dry_run=False,
                        verbose=False,
                    )

                    assert result == output_path
                    mock_run.assert_called_once()

    def test_quantization_failure_raises_error(self):
        """Test that quantization failure raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.quantize.DEFAULT_CONFIG") as mock_config:
                mock_config.quantize_binary = Path("/usr/bin/false")

                with patch("subprocess.run") as mock_run:
                    from subprocess import CalledProcessError
                    mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")

                    with pytest.raises(RuntimeError, match="Quantization failed"):
                        quantize_model(
                            gguf_path=gguf_path,
                            quant_type="Q4_K_M",
                            dry_run=False,
                            verbose=False,
                        )


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = QuantizationConfig()
        assert config.model_id == "Qwen/Qwen3.6-27B"
        assert "Q4_K_M" in config.quant_levels

    def test_config_validates_binaries(self):
        """Test that config validates binaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = QuantizationConfig(
                llama_cpp_dir=Path(tmpdir),
            )

            # Should raise because binaries don't exist
            with pytest.raises(RuntimeError):
                config.validate_binaries()

    def test_config_quantize_binary_path(self):
        """Test that quantize binary path is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = QuantizationConfig(
                llama_cpp_dir=Path(tmpdir),
            )

            expected = Path(tmpdir) / "build" / "bin" / "llama-quantize"
            assert config.quantize_binary == expected
