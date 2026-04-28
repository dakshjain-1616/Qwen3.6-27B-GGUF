"""Command-line interface for qwen36gguf pipeline."""

import sys
from pathlib import Path

import click

from qwen36gguf.config import (
    VALID_QUANT_TYPES,
    QuantizationConfig,
    SmokeTestConfig,
)
from qwen36gguf.pipeline import Pipeline


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing."
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, dry_run: bool) -> None:
    """QWEN36GGUF: Production-ready GGUF quantization pipeline for Qwen3.6-27B."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.option(
    "--model-id",
    default="Qwen/Qwen3.6-27B",
    help="HuggingFace model ID to quantize."
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("out"),
    help="Output directory for quantized models."
)
@click.option(
    "--quant-levels",
    default="Q2_K,Q4_K_M,Q5_K_S,Q8_0",
    help="Comma-separated list of quantization levels."
)
@click.option(
    "--llama-cpp-dir",
    type=click.Path(path_type=Path, exists=True),
    help="Path to llama.cpp directory."
)
@click.option(
    "--hf-repo-id",
    help="HuggingFace repo ID for upload."
)
@click.option(
    "--hf-private",
    is_flag=True,
    help="Make HuggingFace repo private."
)
@click.option(
    "--keep-intermediates",
    is_flag=True,
    help="Keep intermediate files."
)
@click.pass_context
def pipeline(
    ctx: click.Context,
    model_id: str,
    output_dir: Path,
    quant_levels: str,
    llama_cpp_dir: Path | None,
    hf_repo_id: str | None,
    hf_private: bool,
    keep_intermediates: bool,
) -> None:
    """Run the full quantization pipeline."""
    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    # Parse quantization levels
    levels = [q.strip() for q in quant_levels.split(",")]
    invalid = [q for q in levels if q not in VALID_QUANT_TYPES]
    if invalid:
        click.echo(f"Error: Invalid quantization types: {', '.join(invalid)}", err=True)
        click.echo(f"Valid types: {', '.join(VALID_QUANT_TYPES)}", err=True)
        sys.exit(1)

    # Build configuration
    config_kwargs = {
        "model_id": model_id,
        "output_dir": output_dir,
        "quant_levels": levels,
        "hf_repo_id": hf_repo_id,
        "hf_private": hf_private,
        "dry_run": dry_run,
        "verbose": verbose,
        "keep_intermediates": keep_intermediates,
    }

    if llama_cpp_dir:
        config_kwargs["llama_cpp_dir"] = llama_cpp_dir

    try:
        config = QuantizationConfig(**config_kwargs)
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)

    # Run pipeline
    pipeline_runner = Pipeline(config)

    try:
        success = pipeline_runner.run()
        if not success:
            sys.exit(1)
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("out/smoke"),
    help="Output directory for smoke test."
)
@click.option(
    "--quant-levels",
    default=None,
    help="Comma-separated list of quantization levels (overrides default Q2_K,Q4_K_M,Q5_K_S,Q8_0)."
)
@click.pass_context
def smoke(ctx: click.Context, output_dir: Path, quant_levels: str | None) -> None:
    """Run smoke test with TinyLlama model."""
    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    click.echo("=" * 60)
    click.echo("Running smoke test with TinyLlama-1.1B-Chat-v1.0")
    click.echo("=" * 60)

    smoke_kwargs: dict[str, object] = {"output_dir": output_dir}
    if quant_levels is not None:
        levels = [q.strip() for q in quant_levels.split(",") if q.strip()]
        invalid = [q for q in levels if q not in VALID_QUANT_TYPES]
        if invalid:
            click.echo(f"Error: Invalid quantization types: {', '.join(invalid)}", err=True)
            click.echo(f"Valid types: {', '.join(VALID_QUANT_TYPES)}", err=True)
            sys.exit(1)
        smoke_kwargs["quant_levels"] = levels

    config = SmokeTestConfig(**smoke_kwargs)  # type: ignore[arg-type]

    # Create quantization config from smoke config
    quant_config = QuantizationConfig(
        model_id=config.model_id,
        output_dir=config.output_dir,
        quant_levels=config.quant_levels,
        max_benchmark_samples=config.max_benchmark_samples,
        dry_run=dry_run,
        verbose=verbose,
    )

    pipeline_runner = Pipeline(quant_config)

    try:
        success = pipeline_runner.run()
        if success:
            click.echo("\n✓ Smoke test passed!")
            click.echo(f"Output: {output_dir}/MODEL_CARD.md")
        else:
            click.echo("\n✗ Smoke test failed!", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Smoke test failed: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("out"),
    help="Output directory for downloaded model."
)
@click.pass_context
def download(
    ctx: click.Context,
    model_id: str,
    output_dir: Path,
) -> None:
    """Download a model from HuggingFace."""
    from qwen36gguf.download import download_model

    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    try:
        download_model(
            model_id=model_id,
            output_dir=output_dir,
            dry_run=dry_run,
            verbose=verbose,
        )
    except Exception as e:
        click.echo(f"Download failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    help="Output path for GGUF file."
)
@click.option(
    "--outtype",
    default="f16",
    help="Output type (f16, f32, q8_0)."
)
@click.pass_context
def convert(
    ctx: click.Context,
    model_path: Path,
    output_path: Path | None,
    outtype: str,
) -> None:
    """Convert a HuggingFace model to GGUF format."""
    from qwen36gguf.convert import convert_model

    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    try:
        result = convert_model(
            model_path=model_path,
            output_path=output_path,
            outtype=outtype,
            dry_run=dry_run,
            verbose=verbose,
        )
        click.echo(f"Converted model: {result}")
    except Exception as e:
        click.echo(f"Conversion failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("gguf_path", type=click.Path(exists=True, path_type=Path))
@click.argument("quant_type")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    help="Output path for quantized model."
)
@click.pass_context
def quantize(
    ctx: click.Context,
    gguf_path: Path,
    quant_type: str,
    output_path: Path | None,
) -> None:
    """Quantize a GGUF model."""
    from qwen36gguf.quantize import quantize_model

    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    if quant_type not in VALID_QUANT_TYPES:
        click.echo(f"Error: Invalid quantization type: {quant_type}", err=True)
        click.echo(f"Valid types: {', '.join(VALID_QUANT_TYPES)}", err=True)
        sys.exit(1)

    try:
        result = quantize_model(
            gguf_path=gguf_path,
            quant_type=quant_type,
            output_path=output_path,
            dry_run=dry_run,
            verbose=verbose,
        )
        click.echo(f"Quantized model: {result}")
    except Exception as e:
        click.echo(f"Quantization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("gguf_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def benchmark(ctx: click.Context, gguf_path: Path) -> None:
    """Benchmark a GGUF model."""
    from qwen36gguf.bench import benchmark_model

    verbose = ctx.obj.get("verbose", False)

    try:
        results = benchmark_model(
            gguf_path=gguf_path,
            verbose=verbose,
        )
        click.echo("Benchmark Results:")
        for key, value in results.items():
            click.echo(f"  {key}: {value}")
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--bundle-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the assembled hf_export/ folder produced by `pipeline`.",
)
@click.option(
    "--hf-repo-id",
    required=True,
    help="Target HuggingFace repo id, e.g. daksh-neo/qwen36-27b-gguf.",
)
@click.option(
    "--hf-private",
    is_flag=True,
    help="Create the HF repo as private if it doesn't exist yet.",
)
@click.pass_context
def upload(
    ctx: click.Context,
    bundle_dir: Path,
    hf_repo_id: str,
    hf_private: bool,
) -> None:
    """Push an existing hf_export/ bundle to HuggingFace.

    Requires HF_TOKEN with write scope in the environment. Use --dry-run on
    the parent command to preview the file manifest without uploading.
    """
    from qwen36gguf.upload import upload_to_huggingface

    verbose = ctx.obj.get("verbose", False)
    dry_run = ctx.obj.get("dry_run", False)

    try:
        upload_to_huggingface(
            bundle_dir=bundle_dir,
            repo_id=hf_repo_id,
            private=hf_private,
            dry_run=dry_run,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        click.echo(f"Bundle not found: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Upload failed: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
