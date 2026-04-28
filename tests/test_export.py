"""Tests for the hf_export bundle assembly."""

from __future__ import annotations

import tempfile
from pathlib import Path

from qwen36gguf.export import (
    GITATTRIBUTES_LFS,
    assemble_hf_export,
    generate_hf_readme,
)


class TestGenerateHfReadme:
    def test_includes_yaml_frontmatter(self) -> None:
        text = generate_hf_readme(
            model_id="org/model-name",
            benchmark_results={},
            quant_levels=["Q4_K_M"],
        )
        # Frontmatter is delimited by triple-dash on its own lines
        assert text.startswith("---\n")
        assert "license: mit" in text
        assert "base_model: org/model-name" in text
        assert "library_name: gguf" in text

    def test_lists_all_quant_levels(self) -> None:
        text = generate_hf_readme(
            model_id="org/model-name",
            benchmark_results={},
            quant_levels=["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"],
        )
        for q in ("Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"):
            assert f"model-name-{q}.gguf" in text

    def test_uses_llama_server_for_each_quant(self) -> None:
        text = generate_hf_readme(
            model_id="org/model-name",
            benchmark_results={},
            quant_levels=["Q2_K", "Q4_K_M"],
        )
        assert text.count("llama-server -m") == 2
        assert "model-name-Q2_K.gguf" in text
        assert "model-name-Q4_K_M.gguf" in text

    def test_renders_metric_table_when_results_present(self) -> None:
        results = {
            "model-name-Q4_K_M.gguf": {
                "file_size_mb": 600.0,
                "perplexity": 7.5,
                "throughput_tok_per_sec": 42.0,
            },
        }
        text = generate_hf_readme(
            model_id="org/model-name",
            benchmark_results=results,
            quant_levels=["Q4_K_M"],
        )
        # Table headers + the file row
        assert "| Model" in text
        assert "model-name-Q4_K_M.gguf" in text
        # numeric formatting (≤100 → 4 decimals)
        assert "7.5000" in text


class TestAssembleHfExport:
    def test_creates_bundle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            gguf = run_dir / "model-Q4_K_M.gguf"
            gguf.write_bytes(b"\x00\x01\x02 fake gguf")

            bundle = assemble_hf_export(
                output_dir=run_dir,
                model_id="org/model",
                gguf_paths=[gguf],
                benchmark_results={"model-Q4_K_M.gguf": {"file_size_mb": 0.0}},
                quant_levels=["Q4_K_M"],
            )

            assert bundle == run_dir / "hf_export"
            assert bundle.is_dir()
            assert (bundle / "README.md").is_file()
            assert (bundle / ".gitattributes").is_file()
            assert (bundle / "model-Q4_K_M.gguf").is_file()

    def test_gitattributes_has_lfs_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = assemble_hf_export(
                output_dir=Path(tmpdir),
                model_id="org/model",
                gguf_paths=[],
                benchmark_results={},
                quant_levels=[],
            )
            content = (bundle / ".gitattributes").read_text()
            assert content == GITATTRIBUTES_LFS
            assert "*.gguf filter=lfs" in content

    def test_skips_missing_gguf_sources(self) -> None:
        """If a gguf source is missing, it's logged and the rest of the bundle is built."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            real = run_dir / "real-Q4_K_M.gguf"
            real.write_bytes(b"present")
            ghost = run_dir / "ghost-Q2_K.gguf"  # not created

            bundle = assemble_hf_export(
                output_dir=run_dir,
                model_id="org/model",
                gguf_paths=[real, ghost],
                benchmark_results={},
                quant_levels=["Q4_K_M", "Q2_K"],
            )

            assert (bundle / "real-Q4_K_M.gguf").is_file()
            assert not (bundle / "ghost-Q2_K.gguf").exists()

    def test_overwrites_existing_files(self) -> None:
        """Re-running export over an existing bundle replaces the gguf cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            gguf = run_dir / "model-Q4_K_M.gguf"
            gguf.write_bytes(b"v1")

            bundle1 = assemble_hf_export(
                output_dir=run_dir,
                model_id="org/model",
                gguf_paths=[gguf],
                benchmark_results={},
                quant_levels=["Q4_K_M"],
            )
            assert (bundle1 / "model-Q4_K_M.gguf").read_bytes() == b"v1"

            gguf.write_bytes(b"v2")
            bundle2 = assemble_hf_export(
                output_dir=run_dir,
                model_id="org/model",
                gguf_paths=[gguf],
                benchmark_results={},
                quant_levels=["Q4_K_M"],
            )
            assert bundle2 == bundle1
            assert (bundle2 / "model-Q4_K_M.gguf").read_bytes() == b"v2"
