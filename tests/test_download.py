"""Tests for download module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen36gguf.download import download_model


class TestDownloadModel:
    """Tests for download_model function."""

    def test_dry_run_returns_expected_path(self):
        """Test that dry run returns expected path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = download_model(
                model_id="test/model",
                output_dir=output_dir,
                dry_run=True,
                verbose=False,
            )

            assert "test" in str(result).lower() or "model" in str(result).lower()

    @patch("qwen36gguf.download.snapshot_download")
    def test_successful_download(self, mock_snapshot_download):
        """Test successful model download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            expected_path = output_dir / "test-model"
            expected_path.mkdir()

            mock_snapshot_download.return_value = str(expected_path)

            result = download_model(
                model_id="test/model",
                output_dir=output_dir,
                dry_run=False,
                verbose=False,
            )

            assert result == expected_path
            mock_snapshot_download.assert_called_once()

    @patch("qwen36gguf.download.snapshot_download")
    def test_download_failure_raises_error(self, mock_snapshot_download):
        """Test that download failure raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_snapshot_download.side_effect = Exception("Download failed")

            with pytest.raises(RuntimeError, match="Download failed"):
                download_model(
                    model_id="test/model",
                    output_dir=output_dir,
                    dry_run=False,
                    verbose=False,
                )

    @patch("qwen36gguf.download.snapshot_download")
    def test_download_with_verbose(self, mock_snapshot_download):
        """Test download with verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            expected_path = output_dir / "test-model"
            expected_path.mkdir()

            mock_snapshot_download.return_value = str(expected_path)

            result = download_model(
                model_id="test/model",
                output_dir=output_dir,
                dry_run=False,
                verbose=True,
            )

            assert result == expected_path
            mock_snapshot_download.assert_called_once()
