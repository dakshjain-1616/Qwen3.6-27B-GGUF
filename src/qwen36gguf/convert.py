"""Model conversion functionality (HF to GGUF)."""

import logging
import subprocess
from pathlib import Path

from qwen36gguf.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def convert_model(
    model_path: Path,
    output_path: Path | None = None,
    outtype: str = "f16",
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """Convert a HuggingFace model to GGUF format.

    Args:
        model_path: Path to downloaded HuggingFace model.
        output_path: Output path for GGUF file. If None, inferred from model_path.
        outtype: Output type (f16, f32, q8_0).
        dry_run: If True, only log what would be done.
        verbose: Enable verbose logging.

    Returns:
        Path to converted GGUF file.

    Raises:
        RuntimeError: If conversion fails.
    """
    model_path = Path(model_path)

    if output_path is None:
        model_name = model_path.name
        output_path = model_path.parent / f"{model_name}-{outtype}.gguf"
    else:
        output_path = Path(output_path)

    if dry_run:
        logger.info(f"[DRY RUN] Would convert {model_path} to {output_path}")
        return output_path

    # Find convert script
    convert_script = DEFAULT_CONFIG.convert_script
    if not convert_script.exists():
        raise RuntimeError(f"Convert script not found: {convert_script}")

    # Build command
    import sys
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]

    logger.info(f"Converting {model_path.name} to GGUF ({outtype})...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        if verbose and result.stdout:
            logger.debug(result.stdout)

        logger.info(f"Converted to {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(f"Conversion failed: {error_msg}") from e
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}") from e
