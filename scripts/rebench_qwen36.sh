#!/usr/bin/env bash
# One-off re-bench for Qwen3.6-27B Q4_K_M, Q5_K_S, Q8_0 on V100 (16 GB VRAM).
# Pipeline default bench is CPU-only with 300s timeout — too slow for the larger quants.
# Here we partial-offload to GPU sized to a ~14.5 GB safe VRAM budget and bump the timeout.

set -euo pipefail

ROOT="/root/gguf/projects/qwen36-27b-gguf"
PERP="$ROOT/vendor/llama.cpp/build/bin/llama-perplexity"
BENCH="$ROOT/vendor/llama.cpp/build/bin/llama-bench"
WIKI="$ROOT/data/wikitext-2/wiki.test.tokens"
OUT="$ROOT/out/qwen36-27b"
RESULTS="$OUT/rebench"

mkdir -p "$RESULTS"
cd "$ROOT"

# -ngl per quant (65 total layers; 14.5 GB budget; floor on per-layer GB)
declare -A NGL=( [Q4_K_M]=99 [Q5_K_S]=53 [Q8_0]=35 )

for Q in Q4_K_M Q5_K_S Q8_0; do
  GGUF="$OUT/Qwen3.6-27B-$Q.gguf"
  N=${NGL[$Q]}
  echo "=== $Q  (-ngl $N) ==="

  echo "-- perplexity --"
  timeout 1500 "$PERP" \
      -m "$GGUF" -f "$WIKI" \
      --ctx-size 512 -ngl "$N" --threads 8 \
      > "$RESULTS/$Q.perp.stdout" 2> "$RESULTS/$Q.perp.stderr" \
      && echo "perp ok" || echo "perp rc=$?"

  echo "-- throughput --"
  timeout 600 "$BENCH" \
      -m "$GGUF" -p 512 -n 128 -t 4 -ngl "$N" \
      > "$RESULTS/$Q.bench.stdout" 2> "$RESULTS/$Q.bench.stderr" \
      && echo "bench ok" || echo "bench rc=$?"
done

echo "=== DONE ==="
