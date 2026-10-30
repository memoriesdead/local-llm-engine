# Experimental CPU q4 grouped-query attention

This backend contains an actual llama.cpp/GGML CPU engine patch and a numerical
fixture. It is **experimental and unpromoted**. It changes no model weight file,
cache format, model conversion, GPU backend, or sampling rule. No attributable
end-to-end model speedup is established.

Upstream: <https://github.com/ggml-org/llama.cpp>

Required revision: `6050b467c07a95c7b87debd4ddac2f8ba8d9b807`.

The specialization reuses each K row and one V dequantization across six grouped
queries while retaining each query's original online-softmax accumulation order.
It dispatches only for one query, 24 query heads, four KV heads, dimension 256,
q4_0 K/V, at least 512 KV positions, and supported contiguous types/strides and
masks. ALiBi and logit softcap are excluded. Reference mode and other shapes
retain the original path. Existing quantized values remain unchanged; the patch
does not quantize a weight file or select a different model.

## Apply to a dedicated checkout

The package's patch command verifies the revision, patch hash, and target
allowlist. For a manual clean-checkout application:

```bash
BACKEND_DIR=/path/to/local-speed-engine/backends/llama-cpp
CHECKOUT=/path/to/dedicated/llama-cpp-checkout
git -C "$CHECKOUT" rev-parse HEAD
# Require the exact revision above and a clean checkout before continuing.
git -C "$CHECKOUT" apply --check "$BACKEND_DIR/patches/q4-grouped-decode.patch"
git -C "$CHECKOUT" apply "$BACKEND_DIR/patches/q4-grouped-decode.patch"
```

Only `ggml/src/ggml-cpu/ops.cpp` is patched. The independent executable fixture
is supplied in this backend's `tests/` directory and requires no model file.

## Portable Linux CPU build and numerical checks

Requirements: the pinned upstream host-build dependencies, CMake, a compatible
C++ compiler, and OpenMP. This example enables no CUDA and builds only GGML.
Build in a new directory and choose a CPU/thread budget available on your machine.

```bash
BUILD_DIR=/path/to/new/llama-cpu-build
cmake -S "$CHECKOUT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_OPENMP=ON -DGGML_NATIVE=ON -DGGML_CCACHE=OFF
cmake --build "$BUILD_DIR" --target ggml --parallel 2
c++ -O3 -std=c++17 -I "$CHECKOUT/ggml/include" \
  "$BACKEND_DIR/tests/test_q4_attention.cpp" \
  -L "$BUILD_DIR/bin" -Wl,-rpath,"$BUILD_DIR/bin" \
  -lggml-cpu -lggml-base -o "$BUILD_DIR/test-q4-attention"
"$BUILD_DIR/test-q4-attention" 512 1 0 0 5090 1
"$BUILD_DIR/test-q4-attention" 1025 2 1 0 5091 32
"$BUILD_DIR/test-q4-attention" 512 1 2 0 5092 32
```

Fixture arguments are KV length, threads, mask mode (0 dense, 1 masked tail with
sinks, 2 all masked), minimum speed ratio, deterministic seed, and mask columns.
Use minimum speed ratio zero for numerical checks; timing is reported without
requiring a machine-specific speed assertion. The gates require finite output,
maximum absolute error at most 5e-5, and normalized squared error at most 1e-6.
The oracle is actual GGML `use_ref=true` dispatch over the same q4 tensors.

A larger optional check, after budgeting approximately 1.15 GB for its K/V
tensors plus process overhead, is:

```bash
"$BUILD_DIR/test-q4-attention" 1000000 4 1 0 5092 32
```

This is one synthetic attention layer, not one million input tokens through a
language model. On other platforms follow upstream CPU build instructions and
adapt the fixture's link command; no native Windows fixture build is certified.

## Measured scope and limitations

Fifteen historical numerical cases observed zero output difference, including
three seeds, irregular lengths, masked tails, sinks, all-masked inputs, padded
masks, and one million KV positions. Million-position single-layer measurements
were approximately 2.07x to 2.61x faster than the reference on the test host.
Each path used three warm calls; reference calls preceded candidate calls, and
workstation sharing limits timing interpretation.

A full-model retrieval pair measured 6.666 to 6.913 decode tokens per second,
but actual CPU decode placement was unresolved and the candidate remained
unpromoted. That pair does not establish an attributable model-speed gain.
CPU-resident KV alone does not force CPU Flash Attention in the pinned engine.
Use `GGML_Q4_GQA_TRACE=1` when diagnosing whether this specialized CPU route runs.

## Licensing

`LICENSE` contains the complete upstream MIT license and attribution. The patch
retains upstream context and identifies the added specialization; the included
CPU fixture is also supplied under MIT. Preserve the license when redistributing
the source. No model weights, private build state, or dependency binaries are
included.
