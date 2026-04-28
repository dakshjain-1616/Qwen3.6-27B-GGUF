#!/usr/bin/env bash
# Perplexity-only re-run for Qwen3.6-27B Q4_K_M, Q5_K_S, Q8_0.
# First pass OOM'd because llama-perplexity's auto-fit set n_seq=4, blowing
# VRAM budget on partial offload. Here we pin --parallel 1 + --fit off and
# reduce -ngl to leave headroom for KV cache + compute buffers.

set -euo pipefail

ROOT="/root/gguf/projects/qwen36-27b-gguf"
PERP="$ROOT/vendor/llama.cpp/build/bin/llama-perplexity"
WIKI="$ROOT/data/wikitext-2/wiki.test.tokens"
OUT="$ROOT/out/qwen36-27b"
RESULTS="$OUT/rebench"

cd "$ROOT"

# Keep ~3 GB margin for KV cache + compute buffers (n_seq=1, ctx=512).
declare -A NGL=( [Q4_K_M]=50 [Q5_K_S]=42 [Q8_0]=28 )

for Q in Q4_K_M Q5_K_S Q8_0; do
  GGUF="$OUT/Qwen3.6-27B-$Q.gguf"
  N=${NGL[$Q]}
  echo "=== $Q  (-ngl $N --parallel 1 --fit off) ==="
  timeout 1500 "$PERP" \
      -m "$GGUF" -f "$WIKI" \
      --ctx-size 512 -ngl "$N" --threads 8 \
      --parallel 1 --fit off \
      > "$RESULTS/$Q.perp2.stdout" 2> "$RESULTS/$Q.perp2.stderr" \
      && echo "perp ok" || echo "perp rc=$?"
done

echo "=== DONE ==="
