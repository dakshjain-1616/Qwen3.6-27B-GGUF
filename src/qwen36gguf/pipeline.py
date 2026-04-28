"""Pipeline orchestrator for GGUF quantization workflow."""

import logging
from pathlib import Path
from typing import Any

from qwen36gguf.bench import benchmark_model
from qwen36gguf.card import generate_model_card
from qwen36gguf.config import QuantizationConfig
from qwen36gguf.convert import convert_model
from qwen36gguf.download import download_model
from qwen36gguf.export import assemble_hf_export
from qwen36gguf.quantize import quantize_model
from qwen36gguf.upload import upload_to_huggingface

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full GGUF quantization pipeline."""

    def __init__(self, config: QuantizationConfig) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Quantization configuration.
        """
        self.config = config
        self.results: dict[str, Any] = {}

        # Setup logging
        level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

        logger.info(f"Pipeline initialized for {config.model_id}")

    def run(self) -> bool:
        """Execute the full pipeline.

        Returns:
            True if pipeline completed successfully.
        """
        try:
            if self.config.dry_run:
                logger.info("DRY RUN MODE - No actual operations will be performed")

            # Step 1: Validate binaries
            logger.info("Step 1/6: Validating llama.cpp binaries...")
            self._validate_binaries()

            # Step 2: Download model
            logger.info("Step 2/6: Downloading model...")
            model_path = self._download()

            # Step 3: Convert to GGUF
            logger.info("Step 3/6: Converting to GGUF...")
            gguf_path = self._convert(model_path)

            # Step 4: Quantize
            logger.info("Step 4/6: Quantizing...")
            quantized_paths = self._quantize(gguf_path)

            # Step 5: Benchmark
            logger.info("Step 5/6: Benchmarking...")
            benchmark_results = self._benchmark(quantized_paths)

            # Step 6: Generate model card
            logger.info("Step 6/7: Generating model card...")
            self._generate_card(benchmark_results)

            # Step 7: Assemble hf_export bundle (ready-to-push HF repo folder)
            logger.info("Step 7/7: Assembling hf_export bundle...")
            bundle_dir = self._assemble_export(quantized_paths, benchmark_results)
            self.results["hf_export"] = bundle_dir

            if self.config.hf_repo_id:
                logger.info("Uploading hf_export to HuggingFace...")
                self._upload(bundle_dir)

            logger.info("Pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self.config.verbose:
                raise
            return False

    def _validate_binaries(self) -> None:
        """Validate that required binaries exist."""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would validate binaries")
            return

        self.config.validate_binaries()
        logger.info("✓ Binaries validated")

    def _download(self) -> Path:
        """Download model from HuggingFace.

        Returns:
            Path to downloaded model.
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would download {self.config.model_id}")
            return Path("dry_run/model")

        output_dir = self.config.output_dir / "hf_model"
        model_path = download_model(
            model_id=self.config.model_id,
            output_dir=output_dir,
            dry_run=self.config.dry_run,
            verbose=self.config.verbose,
        )
        logger.info(f"✓ Downloaded to {model_path}")
        return model_path

    def _convert(self, model_path: Path) -> Path:
        """Convert model to GGUF format.

        Args:
            model_path: Path to downloaded model.

        Returns:
            Path to GGUF file.
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would convert {model_path}")
            return Path("dry_run/model-f16.gguf")

        output_path = self.config.output_dir / f"{self.config.model_id.split('/')[-1]}-f16.gguf"
        gguf_path = convert_model(
            model_path=model_path,
            output_path=output_path,
            outtype="f16",
            dry_run=self.config.dry_run,
            verbose=self.config.verbose,
        )
        logger.info(f"✓ Converted to {gguf_path}")
        return gguf_path

    def _quantize(self, gguf_path: Path) -> list[Path]:
        """Quantize GGUF model to multiple levels.

        Args:
            gguf_path: Path to GGUF file.

        Returns:
            List of paths to quantized models.
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would quantize {gguf_path}")
            return [Path(f"dry_run/model-{q}.gguf") for q in self.config.quant_levels]

        quantized_paths = []
        for quant_type in self.config.quant_levels:
            output_path = self.config.output_dir / f"{self.config.model_id.split('/')[-1]}-{quant_type}.gguf"
            quantized_path = quantize_model(
                gguf_path=gguf_path,
                quant_type=quant_type,
                output_path=output_path,
                dry_run=self.config.dry_run,
                verbose=self.config.verbose,
            )
            quantized_paths.append(quantized_path)
            logger.info(f"✓ Quantized to {quant_type}: {quantized_path}")

        return quantized_paths

    def _benchmark(self, quantized_paths: list[Path]) -> dict[str, Any]:
        """Benchmark quantized models.

        Args:
            quantized_paths: List of paths to quantized models.

        Returns:
            Dictionary of benchmark results.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Would benchmark models")
            return {}

        results = {}
        for path in quantized_paths:
            result = benchmark_model(
                gguf_path=path,
                verbose=self.config.verbose,
            )
            results[path.name] = result
            logger.info(f"✓ Benchmarked {path.name}")

        return results

    def _generate_card(self, benchmark_results: dict[str, Any]) -> Path:
        """Generate model card.

        Args:
            benchmark_results: Dictionary of benchmark results.

        Returns:
            Path to generated model card.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Would generate model card")
            return Path("dry_run/MODEL_CARD.md")

        card_path = generate_model_card(
            model_id=self.config.model_id,
            output_dir=self.config.output_dir,
            benchmark_results=benchmark_results,
            quant_levels=self.config.quant_levels,
        )
        logger.info(f"✓ Generated model card: {card_path}")
        return card_path

    def _assemble_export(
        self,
        quantized_paths: list[Path],
        benchmark_results: dict[str, Any],
    ) -> Path:
        """Assemble the hf_export bundle.

        Args:
            quantized_paths: Quantized GGUF paths produced by this run.
            benchmark_results: Per-file benchmark metric dicts.

        Returns:
            Path to the assembled ``hf_export/`` directory.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Would assemble hf_export bundle")
            return self.config.output_dir / "hf_export"

        bundle_dir = assemble_hf_export(
            output_dir=self.config.output_dir,
            model_id=self.config.model_id,
            gguf_paths=quantized_paths,
            benchmark_results=benchmark_results,
            quant_levels=self.config.quant_levels,
        )
        logger.info(f"✓ Assembled hf_export at {bundle_dir}")
        return bundle_dir

    def _upload(self, bundle_dir: Path) -> None:
        """Upload the hf_export bundle to HuggingFace.

        Args:
            bundle_dir: Path to the assembled hf_export folder.
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would upload {bundle_dir} to {self.config.hf_repo_id}")
            return

        if not self.config.hf_repo_id:
            logger.info("No HF repo ID configured, skipping upload")
            return

        upload_to_huggingface(
            bundle_dir=bundle_dir,
            repo_id=self.config.hf_repo_id,
            private=self.config.hf_private,
            dry_run=self.config.dry_run,
            verbose=self.config.verbose,
        )
        logger.info(f"✓ Uploaded to {self.config.hf_repo_id}")
