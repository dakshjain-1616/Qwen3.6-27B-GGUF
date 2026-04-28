"""Tests for CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qwen36gguf.cli import cli


class TestCliMain:
    """Tests for main CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "GGUF quantization pipeline" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0

    def test_smoke_command_help(self):
        """Test smoke command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["smoke", "--help"])

        assert result.exit_code == 0
        assert "smoke" in result.output.lower()

    def test_pipeline_command_help(self):
        """Test pipeline command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "--help"])

        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()

    def test_download_command_help(self):
        """Test download command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "--help"])

        assert result.exit_code == 0
        assert "download" in result.output.lower()

    def test_convert_command_help(self):
        """Test convert command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])

        assert result.exit_code == 0
        assert "convert" in result.output.lower()

    def test_quantize_command_help(self):
        """Test quantize command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["quantize", "--help"])

        assert result.exit_code == 0
        assert "quantize" in result.output.lower()

    def test_benchmark_command_help(self):
        """Test benchmark command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()


class TestCliSmokeCommand:
    """Tests for smoke command."""

    @patch("qwen36gguf.cli.Pipeline")
    def test_smoke_command_runs(self, mock_pipeline_class):
        """Test that smoke command runs pipeline."""
        runner = CliRunner()

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = True
        mock_pipeline_class.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "smoke",
                "--output-dir", tmpdir,
            ])

            assert result.exit_code == 0
            mock_pipeline_class.assert_called_once()
            mock_pipeline.run.assert_called_once()

    @patch("qwen36gguf.cli.Pipeline")
    def test_smoke_command_failure(self, mock_pipeline_class):
        """Test smoke command handles failure."""
        runner = CliRunner()

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = False
        mock_pipeline_class.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "smoke",
                "--output-dir", tmpdir,
            ])

            assert result.exit_code != 0


class TestCliPipelineCommand:
    """Tests for pipeline command."""

    def test_pipeline_command_help_shows_options(self):
        """Test that pipeline command help shows all options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "--help"])

        assert result.exit_code == 0
        assert "--model-id" in result.output
        assert "--output-dir" in result.output
        assert "--quant-levels" in result.output


class TestCliQuantizeCommand:
    """Tests for quantize command."""

    def test_quantize_command_invalid_type(self):
        """Test quantize command with invalid quant type."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "model.gguf"
            gguf_file.write_bytes(b"dummy")

            result = runner.invoke(cli, [
                "quantize",
                str(gguf_file),
                "INVALID_TYPE",
            ])

            assert result.exit_code != 0
            assert "Invalid quantization type" in result.output or "Error" in result.output


class TestCliBenchmarkCommand:
    """Tests for benchmark command."""

    @patch("qwen36gguf.bench.benchmark_model")
    def test_benchmark_command_runs(self, mock_benchmark):
        """Test that benchmark command runs."""
        runner = CliRunner()

        mock_benchmark.return_value = {
            "model": "test.gguf",
            "file_size_mb": 100.0,
            "perplexity": 12.34,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "model.gguf"
            gguf_file.write_bytes(b"dummy content" * 1000)

            result = runner.invoke(cli, [
                "benchmark",
                str(gguf_file),
            ])

            assert result.exit_code == 0
            mock_benchmark.assert_called_once()
