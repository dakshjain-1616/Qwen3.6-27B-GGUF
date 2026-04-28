"""Tests for upload module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen36gguf.upload import upload_to_huggingface


def _make_bundle(tmp_root: Path) -> Path:
    """Build a minimal hf_export bundle on disk for upload tests."""
    bundle = tmp_root / "hf_export"
    bundle.mkdir()
    (bundle / "README.md").write_text("# fake card")
    (bundle / ".gitattributes").write_text("*.gguf filter=lfs diff=lfs merge=lfs -text\n")
    (bundle / "model-Q4_K_M.gguf").write_bytes(b"dummy gguf bytes")
    return bundle


class TestUploadToHuggingface:
    """Tests for upload_to_huggingface function."""

    def test_missing_bundle_raises(self) -> None:
        """A non-existent bundle dir surfaces FileNotFoundError, not a hub call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "does-not-exist"
            with pytest.raises(FileNotFoundError, match="hf_export bundle not found"):
                upload_to_huggingface(
                    bundle_dir=missing,
                    repo_id="user/repo",
                    dry_run=False,
                )

    def test_dry_run_does_not_upload(self) -> None:
        """Dry-run lists files and never instantiates HfApi."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _make_bundle(Path(tmpdir))
            with patch("qwen36gguf.upload.HfApi") as mock_hf_api:
                upload_to_huggingface(
                    bundle_dir=bundle,
                    repo_id="user/repo",
                    dry_run=True,
                )
                mock_hf_api.assert_not_called()

    @patch("qwen36gguf.upload.HfApi")
    @patch("qwen36gguf.upload.create_repo")
    def test_upload_calls_upload_folder(
        self, mock_create_repo: MagicMock, mock_hf_api: MagicMock
    ) -> None:
        """Real upload path uses HfApi.upload_folder once with the bundle path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _make_bundle(Path(tmpdir))
            mock_api_instance = MagicMock()
            mock_hf_api.return_value = mock_api_instance

            upload_to_huggingface(
                bundle_dir=bundle,
                repo_id="user/repo",
                dry_run=False,
            )

            mock_create_repo.assert_called_once_with(
                "user/repo", private=False, exist_ok=True, repo_type="model"
            )
            mock_api_instance.upload_folder.assert_called_once_with(
                folder_path=str(bundle),
                repo_id="user/repo",
                repo_type="model",
            )

    @patch("qwen36gguf.upload.HfApi")
    @patch("qwen36gguf.upload.create_repo")
    def test_upload_private_repo(
        self, mock_create_repo: MagicMock, mock_hf_api: MagicMock
    ) -> None:
        """`private=True` is forwarded to create_repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _make_bundle(Path(tmpdir))
            mock_hf_api.return_value = MagicMock()

            upload_to_huggingface(
                bundle_dir=bundle,
                repo_id="user/repo",
                private=True,
                dry_run=False,
            )

            mock_create_repo.assert_called_once_with(
                "user/repo", private=True, exist_ok=True, repo_type="model"
            )

    @patch("qwen36gguf.upload.HfApi")
    @patch("qwen36gguf.upload.create_repo")
    def test_upload_failure_raises_runtime(
        self, mock_create_repo: MagicMock, mock_hf_api: MagicMock
    ) -> None:
        """Upstream hub exceptions become RuntimeError with context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _make_bundle(Path(tmpdir))
            mock_api_instance = MagicMock()
            mock_api_instance.upload_folder.side_effect = Exception("API Error")
            mock_hf_api.return_value = mock_api_instance

            with pytest.raises(RuntimeError, match="Upload to user/repo failed"):
                upload_to_huggingface(
                    bundle_dir=bundle,
                    repo_id="user/repo",
                    dry_run=False,
                )
