# NInfer exact-bit dense embedding gather

This backend contains a C++/CUDA engine patch, its numerical regression tests,
and a gather microbenchmark. It changes no model weight file and performs no
model conversion, downloading, serving, or inference setup.

Upstream: <https://github.com/iamwavecut/ninfer-all>

Required revision: `33163f05fb54459c19e2574607f5d252f4be2ba4`.

The fast route stages independent 16-byte reads before writing a dense BF16
embedding row. It copies raw bits, including signed zero and NaN payloads.
Dispatch requires width 5120, 16-byte-aligned source/output, and 32 through 128
tokens. Other shapes retain the original route. Compatible tensor shapes do
not imply support for an arbitrary model architecture; use the upstream
engine's supported models and artifact formats.

## Apply to a dedicated checkout

The package's patch command verifies the revision, patch hash, and target
allowlist. Manual application to a clean checkout is also possible:

```bash
BACKEND_DIR=/path/to/local-speed-engine/backends/ninfer
CHECKOUT=/path/to/dedicated/ninfer-checkout
git -C "$CHECKOUT" rev-parse HEAD
# Require the exact revision above and a clean checkout before continuing.
git -C "$CHECKOUT" apply --check "$BACKEND_DIR/patches/dense-bf16-gather.patch"
git -C "$CHECKOUT" apply "$BACKEND_DIR/patches/dense-bf16-gather.patch"
```

The patch changes four existing source/test files and adds
`tests/ops/embedding_dense_sweep.cu`. It does not contain a converter, checkpoint,
template, runtime profile, model request, or generated model output.

## Build and verify locally

Requirements: CMake 3.28 or newer, Ninja, a compatible NVIDIA CUDA toolkit and
driver, a C++20 host compiler, FFmpeg development libraries, and curl 7.85 or
newer. The pinned upstream Linux guide uses GCC 13; consult its build guides for
other platforms. The RTX 5090 native route requires CUDA 13.1 or newer. Supply
your own toolkit path; no private environment or prebuilt binary is required.

```bash
CUDA_ROOT=/path/to/cuda
BUILD_DIR=/path/to/new/ninfer-build
cmake -S "$CHECKOUT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_SM120_NATIVE=ON \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build "$BUILD_DIR" --target ninfer-serve quasar_embedding_test --parallel 2
```

After checking that the GPU is available for this work, run the independent CPU
byte-gather oracle against the CUDA result:

```bash
"$BUILD_DIR/tests/quasar_embedding_test" --dense-only
```

Tests cover device and mapped-host tables, dispatch boundaries, repeated IDs,
misalignment, exact BF16 bits, unchanged inputs, allocation guards, and graph
replay with changed IDs. Exit 77 means CUDA was unavailable; it is not a passing
GPU qualification. Full model loading and serving remain upstream operations.

An optional direct kernel benchmark needs no model:

```bash
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch=sm_120a \
  -I "$CHECKOUT/src" -I "$CHECKOUT/include" \
  "$CHECKOUT/tests/ops/embedding_dense_sweep.cu" \
  -o "$BUILD_DIR/embedding_dense_sweep"
"$BUILD_DIR/embedding_dense_sweep" --graph --iterations 8192
```

For other GPUs use the pinned upstream architecture/toolkit instructions and
requalify the route. This package does not claim their performance was measured.

## Measured scope

Historical RTX 5090 hot graph kernel medians, three alternating trials:

| Storage | Tokens | Original, microseconds | Staged, microseconds | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Device | 32 | 1.5566 | 1.1683 | 1.3324x |
| Device | 128 | 3.2325 | 1.2898 | 2.5061x |
| Mapped host | 32 | 11.2670 | 11.2132 | 1.0048x |
| Mapped host | 128 | 34.8476 | 34.4807 | 1.0106x |

These are repeated-input CUDA-event gather timings, not whole-model tokens per
second. Eager mapped-host batch 32 regressed slightly (0.9858x); small batches
and large host batches keep the fallback. Shared workstation conditions and
the small trial count limit interpretation. No general inference speedup or
quality improvement follows from these microbenchmarks.

## Licensing

`LICENSE` contains the complete upstream Apache-2.0 license. The adapted gather
also carries the complete mkyt MIT attribution in `THIRD_PARTY_NOTICES.md`.
Modified source files identify the export's changes. Preserve both notices when
redistributing this patch or its implementation. Dependencies remain in the
upstream checkout under their own notices; no dependency binaries are bundled.
