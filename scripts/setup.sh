#!/bin/bash
# Setup script for llama.cpp with CUDA support
# Clones at a pinned SHA for reproducibility

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="${PROJECT_ROOT}/vendor/llama.cpp"

# Pinned SHA for reproducibility (llama.cpp stable version)
LLAMA_CPP_SHA="0f1bb602dd52d3c0c07ac29c8898f2c58c3fa9b9"

echo "=== Setting up llama.cpp for GGUF quantization ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Target SHA: ${LLAMA_CPP_SHA}"

# Create vendor directory
mkdir -p "${PROJECT_ROOT}/vendor"

# Clone or update llama.cpp
if [ -d "${LLAMA_CPP_DIR}" ]; then
    echo "llama.cpp directory exists, checking SHA..."
    cd "${LLAMA_CPP_DIR}"
    CURRENT_SHA=$(git rev-parse HEAD)
    if [ "$CURRENT_SHA" != "$LLAMA_CPP_SHA" ]; then
        echo "SHA mismatch. Fetching and checking out correct SHA..."
        git fetch origin
        git checkout "${LLAMA_CPP_SHA}"
    else
        echo "SHA matches, no update needed."
    fi
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
    cd "${LLAMA_CPP_DIR}"
    git checkout "${LLAMA_CPP_SHA}"
fi

echo "=== Building llama.cpp with CUDA support ==="
cd "${LLAMA_CPP_DIR}"

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA detected: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
    CMAKE_ARGS="-DLLAMA_CUDA=ON"
else
    echo "WARNING: CUDA not detected. Building without CUDA support."
    CMAKE_ARGS=""
fi

# Clean build
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. ${CMAKE_ARGS} -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF

# Build with all available cores
echo "Building with $(nproc) cores..."
cmake --build . --config Release -j$(nproc)

echo "=== Build complete ==="
echo "Binaries location: ${LLAMA_CPP_DIR}/build/bin"

# Verify binaries exist
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
CONVERT_BIN="${LLAMA_CPP_DIR}/build/bin/llama-convert-hf-to-gguf"

if [ -f "${QUANTIZE_BIN}" ]; then
    echo "✓ llama-quantize: ${QUANTIZE_BIN}"
    ls -lh "${QUANTIZE_BIN}"
else
    echo "ERROR: llama-quantize not found!"
    exit 1
fi

# Check for convert script (may be Python script, not binary)
CONVERT_PY="${LLAMA_CPP_DIR}/convert-hf-to-gguf.py"
if [ -f "${CONVERT_PY}" ]; then
    echo "✓ convert-hf-to-gguf.py: ${CONVERT_PY}"
fi

echo "=== Setup complete ==="
echo "Add to your PATH or use the full paths:"
echo "  export LLAMA_QUANTIZE='${QUANTIZE_BIN}'"
