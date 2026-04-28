"""Model quantization functionality."""

import logging
import subprocess
from pathlib import Path

from qwen36gguf.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def quantize_model(
    gguf_path: Path,
    quant_type: str,
    output_path: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """Quantize a GGUF model to a specific quantization level.

    Args:
        gguf_path: Path to GGUF file to quantize.
        quant_type: Quantization type (e.g., Q4_K_M, Q5_K_M).
        output_path: Output path for quantized model. If None, inferred from input.
        dry_run: If True, only log what would be done.
        verbose: Enable verbose logging.

    Returns:
        Path to quantized GGUF file.

    Raises:
        RuntimeError: If quantization fails.
    """
    gguf_path = Path(gguf_path)

    if output_path is None:
        # Replace extension or add quant type
        stem = gguf_path.stem
        if "-f16" in stem or "-f32" in stem:
            stem = stem.replace("-f16", "").replace("-f32", "")
        output_path = gguf_path.parent / f"{stem}-{quant_type}.gguf"
    else:
        output_path = Path(output_path)

    if dry_run:
        logger.info(f"[DRY RUN] Would quantize {gguf_path} to {quant_type} at {output_path}")
        return output_path

    # Find quantize binary
    quantize_binary = DEFAULT_CONFIG.quantize_binary
    if not quantize_binary.exists():
        raise RuntimeError(f"Quantize binary not found: {quantize_binary}")

    # Build command
    cmd = [
        str(quantize_binary),
        str(gguf_path),
        str(output_path),
        quant_type,
    ]

    logger.info(f"Quantizing {gguf_path.name} to {quant_type}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        if verbose and result.stdout:
            logger.debug(result.stdout)

        logger.info(f"Quantized to {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(f"Quantization failed: {error_msg}") from e
    except Exception as e:
        raise RuntimeError(f"Quantization failed: {e}") from e
