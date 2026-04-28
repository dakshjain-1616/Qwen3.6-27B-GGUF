"""Tests for benchmarking module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen36gguf.bench import (
    BinaryNotFoundError,
    PerplexityError,
    _measure_memory,
    _parse_perplexity,
    _parse_throughput,
    _run_perplexity,
    _run_throughput,
    benchmark_model,
)


class TestBenchmarkModel:
    """Tests for benchmark_model function."""

    def test_benchmark_returns_expected_keys(self):
        """Test that benchmark returns expected result keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-Q4_K_M.gguf"
            gguf_path.write_bytes(b"dummy content" * 1000)  # Make it non-empty

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/nonexistent")
                mock_config.benchmark_binary = Path("/nonexistent")

                results = benchmark_model(
                    gguf_path=gguf_path,
                    verbose=False,
                )

                assert "model" in results
                assert "file_size_mb" in results
                assert results["model"] == "model-Q4_K_M.gguf"
                assert results["file_size_mb"] > 0

    def test_benchmark_with_perplexity(self):
        """Test benchmark with perplexity measurement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-Q4_K_M.gguf"
            gguf_path.write_bytes(b"dummy content" * 1000)

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/usr/bin/echo")
                mock_config.benchmark_binary = Path("/nonexistent")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="final perplexity: 12.34",
                        stderr="",
                        returncode=0,
                    )

                    results = benchmark_model(
                        gguf_path=gguf_path,
                        verbose=False,
                    )

                    assert "perplexity" in results
                    assert results["perplexity"] == 12.34

    def test_benchmark_with_throughput(self):
        """Test benchmark with throughput measurement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-Q4_K_M.gguf"
            gguf_path.write_bytes(b"dummy content" * 1000)

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/nonexistent")
                mock_config.benchmark_binary = Path("/usr/bin/echo")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="128.5 tok/s",
                        stderr="",
                        returncode=0,
                    )

                    results = benchmark_model(
                        gguf_path=gguf_path,
                        verbose=False,
                    )

                    assert "throughput_tok_per_sec" in results
                    assert results["throughput_tok_per_sec"] == 128.5


class TestMeasureMemory:
    """Tests for memory measurement."""

    def test_measure_memory_returns_values(self):
        """Test that memory measurement returns expected values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"x" * 1024 * 1024)  # 1MB file

            results = _measure_memory(gguf_path)

            assert "memory_mb" in results
            assert "baseline_memory_mb" in results
            assert results["memory_mb"] > 0
            assert results["baseline_memory_mb"] > 0

    def test_memory_estimate_based_on_file_size(self):
        """Test that memory estimate scales with file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files of different sizes
            small_file = Path(tmpdir) / "small.gguf"
            large_file = Path(tmpdir) / "large.gguf"

            small_file.write_bytes(b"x" * 1024 * 1024)  # 1MB
            large_file.write_bytes(b"x" * 10 * 1024 * 1024)  # 10MB

            small_results = _measure_memory(small_file)
            large_results = _measure_memory(large_file)

            # Large file should have higher memory estimate
            assert large_results["memory_mb"] > small_results["memory_mb"]


class TestRunPerplexity:
    """Tests for perplexity benchmark."""

    def test_perplexity_binary_not_found(self):
        """Test handling when perplexity binary not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/nonexistent")

                with pytest.raises(BinaryNotFoundError):
                    _run_perplexity(gguf_path, verbose=False)

    def test_perplexity_parses_output(self):
        """Test that perplexity is parsed from output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/usr/bin/echo")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Final perplexity: 15.67\n",
                        stderr="",
                        returncode=0,
                    )

                    results = _run_perplexity(gguf_path, verbose=False)

                    # Should parse the perplexity value
                    assert "perplexity" in results
                    assert results["perplexity"] == 15.67

    def test_perplexity_failure_raises_error(self):
        """Test that perplexity failure raises PerplexityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.perplexity_binary = Path("/usr/bin/false")

                with patch("subprocess.run") as mock_run:
                    from subprocess import CalledProcessError
                    mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")

                    with pytest.raises(PerplexityError):
                        _run_perplexity(gguf_path, verbose=False)


class TestRunThroughput:
    """Tests for throughput benchmark."""

    def test_throughput_binary_not_found(self):
        """Test handling when benchmark binary not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.benchmark_binary = Path("/nonexistent")

                with pytest.raises(BinaryNotFoundError):
                    _run_throughput(gguf_path, verbose=False)

    def test_throughput_parses_tok_per_sec(self):
        """Test that throughput parses tok/s from output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_text("dummy")

            with patch("qwen36gguf.bench.DEFAULT_CONFIG") as mock_config:
                mock_config.benchmark_binary = Path("/usr/bin/echo")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Generation: 256 tok/s\n",
                        stderr="",
                        returncode=0,
                    )

                    results = _run_throughput(gguf_path, verbose=False)

                    assert "throughput_tok_per_sec" in results
                    assert results["throughput_tok_per_sec"] == 256.0


class TestParsePerplexity:
    """Tests for perplexity parsing."""

    def test_parse_perplexity_various_formats(self):
        """Parse perplexity from real llama.cpp llama-perplexity output formats."""
        test_cases = [
            # Canonical llama-perplexity emission
            ("Final estimate: PPL = 8.8994 +/- 0.22850", 8.8994),
            # Alternate phrasing some forks use
            ("Final perplexity: 12.34", 12.34),
            # Minimal "PPL = N" line
            ("[1] PPL = 7.55", 7.55),
        ]

        for output, expected in test_cases:
            result = _parse_perplexity(output)
            assert result == expected, f"Failed for: {output!r} (got {result!r})"

    def test_parse_perplexity_rejects_sub_one(self):
        """PPL < 1.0 is impossible — drop noise values like 'perplexity: 0.30 seconds'."""
        assert _parse_perplexity("perplexity: 0.30 seconds per pass") is None

    def test_parse_perplexity_falls_back_to_chunk_progress(self):
        """If only running chunk estimates are present, take the last one."""
        chunks = "[1]5.6856,[2]6.8733,[3]7.2356,[37]8.8879,[38]8.8994,"
        assert _parse_perplexity(chunks) == 8.8994

    def test_parse_perplexity_no_match(self):
        """Test parsing when no perplexity found."""
        result = _parse_perplexity("some random output without numbers")
        assert result is None


class TestParseThroughput:
    """Tests for throughput parsing."""

    def test_parse_throughput_various_formats(self):
        """Test parsing throughput from various output formats."""
        test_cases = [
            ("128.5 tok/s", 128.5),
            ("256 tokens/s", 256.0),
            ("64 t/s", 64.0),
            ("throughput: 100.5", 100.5),
        ]

        for output, expected in test_cases:
            result = _parse_throughput(output)
            assert result == expected, f"Failed for: {output}"

    def test_parse_throughput_no_match(self):
        """Test parsing when no throughput found."""
        result = _parse_throughput("some random output")
        assert result is None
