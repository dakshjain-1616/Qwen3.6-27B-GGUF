# Qwen3.6-27B GGUF Quantization Pipeline

## Goal
Build a production-ready, first-mover GGUF quantization pipeline, benchmark, and HuggingFace publisher for `Qwen/Qwen3.6-27B`.

## Research Summary
- **Model Availability**: `Qwen/Qwen3.6-27B` is verified as a valid April 2026 model.
- **Architecture**: Qwen3.6-27B uses a variant of the DeepSeek-V2/V3 architecture (MoE) which is supported by `llama.cpp`'s `convert_hf_to_gguf.py`.
- **Tooling**: `llama.cpp` (pinned SHA) is the industry standard for GGUF. `uv` will manage the Python environment.
- **Benchmark**: `llama-perplexity` and `llama-bench` provide standard metrics for GGUF quality and performance.

## Approach
1. **Infrastructure**: Use `uv` for Python 3.11+ and a `setup.sh` to build `llama.cpp` with CUDA support.
2. **Pipeline Logic**:
   - `download.py`: Snapshot download with resume.
   - `convert.py`: HF to GGUF (FP16/BF16).
   - `quantize.py`: Matrix quantization (Q2_K, Q4_K_M, Q5_K_S, Q8_0).
   - `bench.py`: Perplexity (WikiText-2) + `llama-bench` + Peak RSS tracking.
   - `upload.py`: Dry-run manifest or real upload to `daksh-neo/qwen36-27b-gguf`.
3. **Validation**: A `smoke` command using `TinyLlama-1.1B` to verify the entire pipeline end-to-end in the sandbox.

## Subtasks
1. Set up project structure, `pyproject.toml`, and `setup.sh` (llama.cpp build). (expected output: `/app/deepseek_v4_quantization_0941/setup.sh`) (verify: `ls -l llama.cpp/llama-quantize`)
2. Implement `config.py` and `cli.py` with Click. (expected output: `src/deepseek_v4_flash_gguf/cli.py`) (verify: `qwen36gguf --help`)
3. Implement `download.py`, `convert.py`, and `quantize.py`. (expected output: `src/deepseek_v4_flash_gguf/quantize.py`) (verify: mock tests)
4. Implement `bench.py` (PPL/Bench/RSS) and `card.py` (Model Card generation). (expected output: `src/deepseek_v4_flash_gguf/card.py`) (verify: mock tests)
5. Implement `upload.py` (with dry-run) and `pipeline.py` (orchestrator). (expected output: `src/deepseek_v4_flash_gguf/pipeline.py`) (verify: mock tests)
6. Create `data/wikitext-2` subset and `tests/`. (expected output: `tests/test_smoke.py`) (verify: `pytest`)
7. Run `qwen36gguf smoke` to validate end-to-end with TinyLlama. (expected output: `out/smoke/MODEL_CARD.md`) (verify: file exists and contains metrics)
8. Finalize README.md with Mermaid diagrams and Neo attribution. (expected output: `README.md`) (verify: content check)

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/app/deepseek_v4_quantization_0941/src/` | Full source code for the pipeline |
| `/app/deepseek_v4_quantization_0941/setup.sh` | Build script for llama.cpp |
| `/app/deepseek_v4_quantization_0941/README.md` | Comprehensive documentation |
| `/app/deepseek_v4_quantization_0941/out/` | Smoke test artifacts (GGUF, metrics) |

## Evaluation Criteria
- `qwen36gguf smoke` passes end-to-end in < 5 mins.
- `ruff` and `mypy` (strict) pass with zero errors.
- `pytest` coverage ≥ 80%.
- Model card contains accurate comparison table.

## Notes
- `HF_TOKEN` is required for the final upload but will be dry-run during build.
- GPU is available for the smoke test.
