"""Configuration module for qwen36gguf pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QuantizationConfig:
    """Configuration for GGUF quantization."""

    # Model settings
    model_id: str = "Qwen/Qwen3.6-27B"
    output_dir: Path = field(default_factory=lambda: Path("out"))

    # Quantization levels to generate
    quant_levels: list[str] = field(default_factory=lambda: ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"])

    # llama.cpp paths
    llama_cpp_dir: Path | None = None

    # Benchmark settings
    benchmark_dataset: str = "wikitext"
    benchmark_config: str = "wikitext-2-raw-v1"
    benchmark_split: str = "test"
    max_benchmark_samples: int = 100

    # HuggingFace settings
    hf_repo_id: str | None = None
    hf_private: bool = False

    # Execution settings
    dry_run: bool = False
    verbose: bool = False
    keep_intermediates: bool = False

    def __post_init__(self) -> None:
        """Resolve paths and validate configuration."""
        if self.llama_cpp_dir is None:
            # Default to vendor/llama.cpp relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.llama_cpp_dir = project_root / "vendor" / "llama.cpp"

        self.llama_cpp_dir = Path(self.llama_cpp_dir)
        self.output_dir = Path(self.output_dir)

        # Validate llama.cpp directory exists
        if not self.llama_cpp_dir.exists():
            raise ValueError(f"llama.cpp directory not found: {self.llama_cpp_dir}")

    @property
    def quantize_binary(self) -> Path:
        """Path to llama-quantize binary."""
        assert self.llama_cpp_dir is not None, "llama_cpp_dir must be set"
        return self.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    @property
    def convert_script(self) -> Path:
        """Path to convert-hf-to-gguf.py script."""
        assert self.llama_cpp_dir is not None, "llama_cpp_dir must be set"
        return self.llama_cpp_dir / "convert_hf_to_gguf.py"

    @property
    def perplexity_binary(self) -> Path:
        """Path to llama-perplexity binary."""
        assert self.llama_cpp_dir is not None, "llama_cpp_dir must be set"
        return self.llama_cpp_dir / "build" / "bin" / "llama-perplexity"

    @property
    def benchmark_binary(self) -> Path:
        """Path to llama-bench binary."""
        assert self.llama_cpp_dir is not None, "llama_cpp_dir must be set"
        return self.llama_cpp_dir / "build" / "bin" / "llama-bench"

    def validate_binaries(self) -> None:
        """Validate that required binaries exist."""
        missing = []

        if not self.quantize_binary.exists():
            missing.append(f"llama-quantize: {self.quantize_binary}")

        if not self.convert_script.exists():
            missing.append(f"convert_hf_to_gguf.py: {self.convert_script}")

        if missing:
            raise RuntimeError(
                "Missing required llama.cpp binaries. "
                f"Run setup.sh first. Missing: {', '.join(missing)}"
            )


@dataclass
class SmokeTestConfig:
    """Configuration for smoke test with TinyLlama."""

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: Path = field(default_factory=lambda: Path("out/smoke"))
    quant_levels: list[str] = field(
        default_factory=lambda: ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]
    )
    max_benchmark_samples: int = 10

    def __post_init__(self) -> None:
        """Resolve paths."""
        self.output_dir = Path(self.output_dir)


# Default configurations
DEFAULT_CONFIG = QuantizationConfig()
SMOKE_CONFIG = SmokeTestConfig()

# Valid quantization types from llama.cpp
VALID_QUANT_TYPES = [
    "Q2_K",
    "Q4_0", "Q4_1", "Q4_K", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "Q8_K", "F16", "F32",
]

# Environment variable overrides
ENV_PREFIX = "QWEN36GGUF_"


def load_config_from_env() -> QuantizationConfig:
    """Load configuration from environment variables."""
    kwargs: dict[str, str | Path | bool | None] = {}

    if model_id := os.getenv(f"{ENV_PREFIX}MODEL_ID"):
        kwargs["model_id"] = model_id

    if output_dir := os.getenv(f"{ENV_PREFIX}OUTPUT_DIR"):
        kwargs["output_dir"] = Path(output_dir)

    if llama_cpp_dir := os.getenv(f"{ENV_PREFIX}LLAMA_CPP_DIR"):
        kwargs["llama_cpp_dir"] = Path(llama_cpp_dir)

    if hf_repo_id := os.getenv(f"{ENV_PREFIX}HF_REPO_ID"):
        kwargs["hf_repo_id"] = hf_repo_id

    if hf_private := os.getenv(f"{ENV_PREFIX}HF_PRIVATE"):
        kwargs["hf_private"] = hf_private.lower() in ("true", "1", "yes")

    return QuantizationConfig(**kwargs)  # type: ignore[arg-type]
