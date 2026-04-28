"""Benchmarking functionality for GGUF models."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import psutil

from qwen36gguf.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BenchError(Exception):
    """Base exception for benchmarking errors."""
    pass


class BinaryNotFoundError(BenchError):
    """Raised when a required binary is not found."""
    pass


class PerplexityError(BenchError):
    """Raised when perplexity benchmark fails."""
    pass


class ThroughputError(BenchError):
    """Raised when throughput benchmark fails."""
    pass


class MemoryMeasurementError(BenchError):
    """Raised when memory measurement fails."""
    pass


def benchmark_model(
    gguf_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Benchmark a GGUF model.

    Runs perplexity, throughput, and memory benchmarks.

    Args:
        gguf_path: Path to GGUF file.
        verbose: Enable verbose logging.

    Returns:
        Dictionary of benchmark results.

    Raises:
        BenchError: If benchmarking fails catastrophically.
    """
    gguf_path = Path(gguf_path)

    results = {
        "model": gguf_path.name,
        "file_size_mb": round(gguf_path.stat().st_size / (1024 * 1024), 2),
    }

    # Run perplexity benchmark
    try:
        perplexity_results = _run_perplexity(gguf_path, verbose)
        results.update(perplexity_results)
    except BinaryNotFoundError:
        logger.warning("Perplexity binary not found, skipping perplexity benchmark")
        results["perplexity"] = None
    except PerplexityError as e:
        logger.warning(f"Perplexity benchmark failed: {e}")
        results["perplexity"] = None

    # Run throughput benchmark
    try:
        throughput_results = _run_throughput(gguf_path, verbose)
        results.update(throughput_results)
    except BinaryNotFoundError:
        logger.warning("Benchmark binary not found, skipping throughput benchmark")
        results["throughput_tok_per_sec"] = None
    except ThroughputError as e:
        logger.warning(f"Throughput benchmark failed: {e}")
        results["throughput_tok_per_sec"] = None

    # Measure memory
    try:
        memory_results = _measure_memory(gguf_path)
        results.update(memory_results)
    except MemoryMeasurementError as e:
        logger.warning(f"Memory measurement failed: {e}")
        results["memory_mb"] = None

    return results


def _get_wikitext_path() -> Path | None:
    """Get the path to the WikiText-2 test data.

    Returns:
        Path to wikitext test file, or None if not found.
    """
    # Look for wikitext in data directory
    project_root = Path(__file__).parent.parent.parent
    wikitext_paths = [
        project_root / "data" / "wikitext-2" / "wiki.test.tokens",
        project_root / "data" / "wikitext-2" / "wikitext-2-raw-v1" / "wiki.test.raw",
        project_root / "data" / "wikitext-2" / "test.txt",
    ]

    for path in wikitext_paths:
        if path.exists():
            return path

    return None


def _run_perplexity(
    gguf_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run perplexity benchmark using llama-perplexity.

    Args:
        gguf_path: Path to GGUF file.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with perplexity results.

    Raises:
        BinaryNotFoundError: If perplexity binary not found.
        PerplexityError: If perplexity calculation fails.
    """
    perplexity_binary = DEFAULT_CONFIG.perplexity_binary

    if not perplexity_binary.exists():
        raise BinaryNotFoundError(f"Perplexity binary not found: {perplexity_binary}")

    # Get wikitext data path
    wikitext_path = _get_wikitext_path()

    if wikitext_path is None:
        logger.warning("WikiText-2 data not found, using empty perplexity calculation")
        # Fall back to minimal context test
        cmd = [
            str(perplexity_binary),
            "-m", str(gguf_path),
            "--ctx-size", "512",
            "--threads", "4",
        ]
    else:
        cmd = [
            str(perplexity_binary),
            "-m", str(gguf_path),
            "-f", str(wikitext_path),
            "--ctx-size", "512",
            "--threads", "4",
        ]

    logger.info(f"Running perplexity benchmark on {gguf_path.name}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise PerplexityError(f"llama-perplexity failed: {result.stderr}")

        # Parse perplexity from output
        output = result.stdout + result.stderr
        perplexity = _parse_perplexity(output)

        if perplexity is None:
            raise PerplexityError("Could not parse perplexity from output")

        return {"perplexity": perplexity}

    except subprocess.TimeoutExpired:
        raise PerplexityError("Perplexity benchmark timed out after 300s") from None
    except subprocess.SubprocessError as e:
        raise PerplexityError(f"Perplexity subprocess error: {e}") from e


def _parse_perplexity(output: str) -> float | None:
    """Parse the final perplexity value from llama-perplexity output.

    The canonical line emitted by llama.cpp's ``llama-perplexity`` is::

        Final estimate: PPL = 8.8994 +/- 0.22850

    Older / alternate builds also use ``Final perplexity: 8.8994`` or
    ``perplexity = 8.8994``. Numbers below 1.0 are rejected because PPL is
    bounded below by 1; in practice anything sub-1 means a regex picked up
    a timing or progress number by mistake.

    Args:
        output: stdout + stderr from llama-perplexity.

    Returns:
        Perplexity value, or None if no plausible PPL was found.
    """
    patterns = (
        r"final\s+estimate\s*[:\s]+ppl\s*=\s*(\d+\.\d+)",
        r"final\s+ppl\s*[:=]\s*(\d+\.\d+)",
        r"final\s+perplexity\s*[:=]\s*(\d+\.\d+)",
        r"\bppl\s*=\s*(\d+\.\d+)",
    )
    output_lower = output.lower()

    for pattern in patterns:
        match = re.search(pattern, output_lower)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            # PPL is mathematically ≥ 1.0; values below that are parser noise.
            if value >= 1.0:
                return value

    # Fallback: scan reverse-order chunk progress like "[38]8.8994," and
    # return the last (highest-index) running estimate. This is robust to
    # builds that don't emit a "Final estimate" header.
    chunk_matches = re.findall(r"\[\d+\](\d+\.\d+)", output)
    if chunk_matches:
        try:
            value = float(chunk_matches[-1])
        except ValueError:
            return None
        if value >= 1.0:
            return value

    return None


def _run_throughput(
    gguf_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run throughput benchmark using llama-bench.

    Args:
        gguf_path: Path to GGUF file.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with throughput results.

    Raises:
        BinaryNotFoundError: If benchmark binary not found.
        ThroughputError: If throughput calculation fails.
    """
    bench_binary = DEFAULT_CONFIG.benchmark_binary

    if not bench_binary.exists():
        raise BinaryNotFoundError(f"Benchmark binary not found: {bench_binary}")

    cmd = [
        str(bench_binary),
        "-m", str(gguf_path),
        "-p", "512",  # prompt tokens
        "-n", "128",  # generation tokens
        "-t", "4",    # threads
    ]

    logger.info(f"Running throughput benchmark on {gguf_path.name}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise ThroughputError(f"llama-bench failed: {result.stderr}")

        # Parse throughput from output
        output = result.stdout + result.stderr
        throughput = _parse_throughput(output)

        if throughput is None:
            raise ThroughputError("Could not parse throughput from output")

        return {"throughput_tok_per_sec": throughput}

    except subprocess.TimeoutExpired:
        raise ThroughputError("Throughput benchmark timed out after 120s") from None
    except subprocess.SubprocessError as e:
        raise ThroughputError(f"Throughput subprocess error: {e}") from e


def _parse_throughput(output: str) -> float | None:
    """Parse the token-generation throughput from llama-bench output.

    llama-bench prints a Markdown-style table whose rows look like::

        | model | size | params | backend | ngl | threads | test | t/s |
        | ...   | ...  | ...    | CUDA    | 99  | 4       | pp32 | 3580.65 ± 12.94 |
        | ...   | ...  | ...    | CUDA    | 99  | 4       | tg16 |  346.78 ± 10.97 |

    The relevant value is the ``tg<N>`` (token-generation) row's mean
    throughput — the number immediately before ``±``.

    Args:
        output: combined stdout + stderr from llama-bench.

    Returns:
        Token-generation throughput (tok/sec) or None if no tg row was found.
    """
    # First try the table row form (most reliable on modern llama-bench).
    for line in output.splitlines():
        # Row separators have only `|` and `-`; skip them.
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # Find any cell starting with "tg" (token generation): tg16, tg32, ...
        if not any(c.startswith("tg") for c in cells):
            continue
        # The last cell holds "<mean> ± <stddev>" — extract the mean.
        last = cells[-1]
        match = re.match(r"\s*(\d+\.?\d*)\s*(?:±|\+/-)", last)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Fallback: literal "tok/s" / "tokens/s" / "t/s" / "throughput: N"
    # forms used by older llama-bench builds and CSV variants.
    fallback_patterns = (
        r"(\d+\.?\d*)\s*tok/s",
        r"(\d+\.?\d*)\s*tokens/s",
        r"(\d+\.?\d*)\s*t/s",
        r"throughput[:\s]+(\d+\.?\d*)",
    )
    for pattern in fallback_patterns:
        match = re.search(pattern, output.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def _measure_memory(gguf_path: Path) -> dict[str, Any]:
    """Measure memory usage for loading the model.

    Args:
        gguf_path: Path to GGUF file.

    Returns:
        Dictionary with memory results.

    Raises:
        MemoryMeasurementError: If memory measurement fails.
    """
    try:
        process = psutil.Process()

        # Get baseline memory
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        # Memory estimate based on file size
        # GGUF models typically use ~1.2-1.5x file size in RAM
        file_size_mb = gguf_path.stat().st_size / (1024 * 1024)
        estimated_mb = file_size_mb * 1.3

        return {
            "memory_mb": round(estimated_mb, 2),
            "baseline_memory_mb": round(baseline_mb, 2),
        }
    except Exception as e:
        raise MemoryMeasurementError(f"Failed to measure memory: {e}") from e
