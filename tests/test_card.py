"""Tests for model card generation module."""

import tempfile
from pathlib import Path

from qwen36gguf.card import generate_markdown_table, generate_model_card


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_model_card_created(self):
        """Test that model card file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark_results = {
                "model-Q4_K_M.gguf": {
                    "model": "model-Q4_K_M.gguf",
                    "file_size_mb": 100.5,
                    "perplexity": 12.34,
                    "throughput_tok_per_sec": 128.5,
                    "memory_mb": 150.0,
                }
            }

            card_path = generate_model_card(
                model_id="test/model",
                output_dir=output_dir,
                benchmark_results=benchmark_results,
                quant_levels=["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"],
            )

            assert card_path.exists()
            assert card_path.name == "MODEL_CARD.md"

    def test_model_card_contains_model_info(self):
        """Test that model card contains model information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark_results = {}

            card_path = generate_model_card(
                model_id="test/model",
                output_dir=output_dir,
                benchmark_results=benchmark_results,
                quant_levels=["Q4_K_M"],
            )

            content = card_path.read_text()
            assert "test/model" in content
            assert "Q4_K_M" in content

    def test_model_card_contains_quant_levels(self):
        """Test that model card contains all quantization levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark_results = {}
            quant_levels = ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]

            card_path = generate_model_card(
                model_id="test/model",
                output_dir=output_dir,
                benchmark_results=benchmark_results,
                quant_levels=quant_levels,
            )

            content = card_path.read_text()
            # Check quant matrix table
            assert "Q2_K" in content
            assert "Q4_K_M" in content
            assert "Q5_K_S" in content
            assert "Q8_0" in content
            # Check descriptions
            assert "Edge/mobile" in content
            assert "balanced quality" in content.lower()

    def test_model_card_contains_benchmark_table(self):
        """Test that model card contains benchmark results table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark_results = {
                "model-Q4_K_M.gguf": {
                    "model": "model-Q4_K_M.gguf",
                    "file_size_mb": 100.5,
                    "perplexity": 12.34,
                }
            }

            card_path = generate_model_card(
                model_id="test/model",
                output_dir=output_dir,
                benchmark_results=benchmark_results,
                quant_levels=["Q4_K_M"],
            )

            content = card_path.read_text()
            assert "Benchmark Results" in content
            assert "file_size_mb" in content or "file size" in content.lower()

    def test_model_card_contains_llama_server_commands(self):
        """Test that model card contains llama-server commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark_results = {}

            card_path = generate_model_card(
                model_id="test/model",
                output_dir=output_dir,
                benchmark_results=benchmark_results,
                quant_levels=["Q4_K_M"],
            )

            content = card_path.read_text()
            assert "llama-server" in content
            assert "--host 0.0.0.0" in content
            assert "--port 8080" in content


class TestGenerateMarkdownTable:
    """Tests for generate_markdown_table function."""

    def test_empty_results(self):
        """Test handling of empty benchmark results."""
        result = generate_markdown_table({})
        assert "No benchmark results" in result

    def test_single_result(self):
        """Test table generation with single result."""
        benchmark_results = {
            "model-Q4_K_M.gguf": {
                "model": "model-Q4_K_M.gguf",
                "file_size_mb": 100.5,
                "perplexity": 12.34,
            }
        }

        result = generate_markdown_table(benchmark_results)
        assert "|" in result  # Markdown table format
        assert "model-Q4_K_M.gguf" in result
        assert "100.5" in result or "100.50" in result

    def test_multiple_results(self):
        """Test table generation with multiple results."""
        benchmark_results = {
            "model-Q2_K.gguf": {
                "model": "model-Q2_K.gguf",
                "file_size_mb": 50.0,
                "perplexity": 15.0,
            },
            "model-Q8_0.gguf": {
                "model": "model-Q8_0.gguf",
                "file_size_mb": 200.0,
                "perplexity": 10.0,
            }
        }

        result = generate_markdown_table(benchmark_results)
        lines = result.split("\n")
        # Should have header, separator, and 2 data rows
        assert len(lines) >= 3
        assert "model-Q2_K.gguf" in result
        assert "model-Q8_0.gguf" in result

    def test_none_values(self):
        """Test handling of None values in results."""
        benchmark_results = {
            "model-Q4_K_M.gguf": {
                "model": "model-Q4_K_M.gguf",
                "file_size_mb": 100.5,
                "perplexity": None,
            }
        }

        result = generate_markdown_table(benchmark_results)
        assert "N/A" in result

    def test_float_formatting(self):
        """Test proper float formatting."""
        benchmark_results = {
            "model-Q4_K_M.gguf": {
                "model": "model-Q4_K_M.gguf",
                "file_size_mb": 100.0,
                "perplexity": 12.345678,
            }
        }

        result = generate_markdown_table(benchmark_results)
        # Should format to reasonable precision
        assert "12.34" in result or "12.345" in result or "12.346" in result
