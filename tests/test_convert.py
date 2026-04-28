"""Tests for convert module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen36gguf.convert import convert_model


class TestConvertModel:
    """Tests for convert_model function."""

    def test_dry_run_returns_expected_path(self):
        """Test that dry run returns expected output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model"
            model_path.mkdir()

            result = convert_model(
                model_path=model_path,
                outtype="f16",
                dry_run=True,
                verbose=False,
            )

            assert result.name == "test-model-f16.gguf"

    def test_custom_output_path(self):
        """Test conversion with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model"
            model_path.mkdir()
            custom_output = Path(tmpdir) / "custom.gguf"

            result = convert_model(
                model_path=model_path,
                output_path=custom_output,
                outtype="f16",
                dry_run=True,
                verbose=False,
            )

            assert result == custom_output

    def test_missing_convert_script_raises_error(self):
        """Test that missing convert script raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model"
            model_path.mkdir()

            with patch("qwen36gguf.convert.DEFAULT_CONFIG") as mock_config:
                mock_config.convert_script = Path("/nonexistent/convert.py")

                with pytest.raises(RuntimeError, match="Convert script not found"):
                    convert_model(
                        model_path=model_path,
                        outtype="f16",
                        dry_run=False,
                        verbose=False,
                    )

    def test_successful_conversion(self):
        """Test successful conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model"
            model_path.mkdir()
            output_path = Path(tmpdir) / "output.gguf"

            with patch("qwen36gguf.convert.DEFAULT_CONFIG") as mock_config:
                mock_config.convert_script = Path("/usr/bin/echo")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="",
                        stderr="",
                        returncode=0,
                    )

                    result = convert_model(
                        model_path=model_path,
                        output_path=output_path,
                        outtype="f16",
                        dry_run=False,
                        verbose=False,
                    )

                    assert result == output_path
                    mock_run.assert_called_once()

    def test_conversion_failure_raises_error(self):
        """Test that conversion failure raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model"
            model_path.mkdir()

            with patch("qwen36gguf.convert.DEFAULT_CONFIG") as mock_config:
                mock_config.convert_script = Path("/usr/bin/false")

                with patch("subprocess.run") as mock_run:
                    from subprocess import CalledProcessError
                    mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Conversion failed")

                    with pytest.raises(RuntimeError, match="Conversion failed"):
                        convert_model(
                            model_path=model_path,
                            outtype="f16",
                            dry_run=False,
                            verbose=False,
                        )

    def test_different_outtypes(self):
        """Test conversion with different output types."""
        outtypes = ["f16", "f32", "q8_0"]

        for outtype in outtypes:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "test-model"
                model_path.mkdir()

                result = convert_model(
                    model_path=model_path,
                    outtype=outtype,
                    dry_run=True,
                    verbose=False,
                )

                assert outtype in result.name
