"""Model download functionality."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_model(
    model_id: str,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """Download a model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0").
        output_dir: Directory to save the model.
        dry_run: If True, only log what would be done.
        verbose: Enable verbose logging.

    Returns:
        Path to downloaded model directory.

    Raises:
        RuntimeError: If download fails.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would download {model_id} to {output_dir}")
        return output_dir / model_id.split("/")[-1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading {model_id}...")

        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,  # type: ignore[call-overload]
        )

        logger.info(f"Downloaded to {local_path}")
        return Path(local_path)

    except Exception as e:
        raise RuntimeError(f"Failed to download {model_id}: {e}") from e
