"""HuggingFace upload functionality.

Uploads the assembled ``hf_export/`` bundle (produced by
:func:`qwen36gguf.export.assemble_hf_export`) to a HuggingFace model repo
using ``HfApi().upload_folder``. The bundle's layout becomes the repo
layout verbatim — no per-file enumeration here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logger = logging.getLogger(__name__)


def upload_to_huggingface(
    bundle_dir: Path,
    repo_id: str,
    private: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Upload the hf_export bundle to a HuggingFace model repo.

    Args:
        bundle_dir: Path to the assembled ``hf_export/`` folder.
        repo_id: Target HuggingFace repo id (e.g. ``"daksh-neo/qwen36-27b-gguf"``).
        private: Create the repo as private if it doesn't already exist.
        dry_run: If True, list what would be uploaded (filename + size) and return.
        verbose: Enable verbose hub logging.

    Raises:
        FileNotFoundError: If ``bundle_dir`` does not exist.
        RuntimeError: If the upload itself fails.
    """
    bundle_dir = Path(bundle_dir)

    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"hf_export bundle not found: {bundle_dir}")

    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file())

    if dry_run:
        logger.info(f"[DRY RUN] Would upload {len(files)} files to {repo_id}:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            rel = f.relative_to(bundle_dir)
            logger.info(f"  {rel}  ({size_mb:.2f} MB)")
        return

    api = HfApi()

    try:
        create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        logger.info(f"Created/verified repo: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo creation warning: {e}")

    logger.info(f"Uploading {len(files)} files from {bundle_dir} to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(bundle_dir),
            repo_id=repo_id,
            repo_type="model",
        )
    except Exception as e:
        raise RuntimeError(f"Upload to {repo_id} failed: {e}") from e

    logger.info(f"Uploaded to https://huggingface.co/{repo_id}")
