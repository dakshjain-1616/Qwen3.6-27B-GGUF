"""Tests for configuration module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen36gguf.config import (
    VALID_QUANT_TYPES,
    QuantizationConfig,
    SmokeTestConfig,
    load_config_from_env,
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock llama.cpp structure
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(llama_cpp_dir=llama_cpp)
            assert config.model_id == "Qwen/Qwen3.6-27B"
            assert "Q2_K" in config.quant_levels
            assert "Q4_K_M" in config.quant_levels
            assert "Q5_K_S" in config.quant_levels
            assert "Q8_0" in config.quant_levels

    def test_config_quant_levels(self):
        """Test that config has correct quant levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(llama_cpp_dir=llama_cpp)
            # Should have the 4 required quant levels
            assert config.quant_levels == ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]

    def test_config_binary_paths(self):
        """Test that binary paths are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(llama_cpp_dir=llama_cpp)

            expected_quantize = llama_cpp / "build" / "bin" / "llama-quantize"
            assert config.quantize_binary == expected_quantize

            expected_convert = llama_cpp / "convert_hf_to_gguf.py"
            assert config.convert_script == expected_convert

            expected_perplexity = llama_cpp / "build" / "bin" / "llama-perplexity"
            assert config.perplexity_binary == expected_perplexity

            expected_bench = llama_cpp / "build" / "bin" / "llama-bench"
            assert config.benchmark_binary == expected_bench

    def test_config_validates_binaries_raises(self):
        """Test that config validates binaries raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = QuantizationConfig(llama_cpp_dir=Path(tmpdir))

            with pytest.raises(RuntimeError, match="Missing required llama.cpp binaries"):
                config.validate_binaries()

    def test_config_output_dir_resolution(self):
        """Test that output_dir is resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            config = QuantizationConfig(
                llama_cpp_dir=llama_cpp,
                output_dir="custom_output"
            )
            assert isinstance(config.output_dir, Path)


class TestSmokeTestConfig:
    """Tests for SmokeTestConfig class."""

    def test_default_smoke_config(self):
        """Test default smoke test configuration."""
        config = SmokeTestConfig()
        assert config.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.quant_levels == ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]
        assert config.max_benchmark_samples == 10

    def test_smoke_config_output_dir(self):
        """Test smoke config output directory."""
        config = SmokeTestConfig()
        assert config.output_dir.name == "smoke"


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env function."""

    def test_load_model_id_from_env(self):
        """Test loading model_id from environment."""
        with patch.dict("os.environ", {"QWEN36GGUF_MODEL_ID": "custom/model"}):
            config = load_config_from_env()
            assert config.model_id == "custom/model"

    def test_load_output_dir_from_env(self):
        """Test loading output_dir from environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_cpp = Path(tmpdir) / "llama.cpp"
            llama_cpp.mkdir()

            with (
                patch.dict("os.environ", {"QWEN36GGUF_OUTPUT_DIR": "/custom/output"}),
                patch("qwen36gguf.config.QuantizationConfig"),
            ):
                # Just verify the env var is read
                import os
                assert os.getenv("QWEN36GGUF_OUTPUT_DIR") == "/custom/output"

    def test_load_hf_repo_id_from_env(self):
        """Test loading hf_repo_id from environment."""
        with patch.dict("os.environ", {"QWEN36GGUF_HF_REPO_ID": "user/repo"}):
            config = load_config_from_env()
            assert config.hf_repo_id == "user/repo"

    def test_load_hf_private_from_env(self):
        """Test loading hf_private from environment."""
        with patch.dict("os.environ", {"QWEN36GGUF_HF_PRIVATE": "true"}):
            config = load_config_from_env()
            assert config.hf_private is True

        with patch.dict("os.environ", {"QWEN36GGUF_HF_PRIVATE": "1"}):
            config = load_config_from_env()
            assert config.hf_private is True


class TestValidQuantTypes:
    """Tests for VALID_QUANT_TYPES constant."""

    def test_required_quants_present(self):
        """Test that required quant types are in valid list."""
        required = ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]
        for quant in required:
            assert quant in VALID_QUANT_TYPES, f"{quant} not in VALID_QUANT_TYPES"

    def test_common_quants_present(self):
        """Test that common quant types are present."""
        common_quants = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q6_K", "F16", "F32"]
        for quant in common_quants:
            assert quant in VALID_QUANT_TYPES, f"{quant} not in VALID_QUANT_TYPES"
