// Local Speed Engine CPU attention fixture. Licensed under MIT; see LICENSE.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// CPU-only, one full-attention layer. Actual GGML graph dispatch; no model or GPU.
int main(int argc, char ** argv) {
    const int64_t n = argc > 1 ? std::atoll(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 4;
    const int mode = argc > 3 ? std::atoi(argv[3]) : 0; // 0 dense, 1 tail mask+sinks, 2 all masked
    const double min_speedup = argc > 4 ? std::atof(argv[4]) : 0;
    const int seed = argc > 5 ? std::atoi(argv[5]) : 5090;
    const int mask_columns = argc > 6 ? std::atoi(argv[6]) : 1;
    if (n < 512 || n > 1000000 || threads < 1 || threads > 4 || mode < 0 || mode > 2 || mask_columns < 1 || mask_columns > 32) return 2;
    constexpr int d = 256, nq = 24, nkv = 4;
    const size_t kv_bytes = ggml_row_size(GGML_TYPE_Q4_0, d) * n * nkv;
    ggml_context * ctx = ggml_init({2 * kv_bytes + n * mask_columns * sizeof(ggml_fp16_t) + 8 * 1024 * 1024, nullptr, false});
    if (!ctx) return 3;
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, nq, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, n, nkv, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, n, nkv, 1);
    auto * mask = mode || mask_columns > 1 ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, mask_columns, 1, 1) : nullptr;
    auto * sinks = mode == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nq) : nullptr;
    for (int h = 0; h < nq; ++h) for (int j = 0; j < d; ++j)
        static_cast<float *>(q->data)[h * d + j] = std::sin(float(h * 37 + j * 17 + seed) * .031f);
    // A deterministic 1,021-row period bounds initialization memory at 1M context.
    std::vector<float> input(1021 * d);
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q4_0, d) * 1021);
    for (int which = 0; which < 2; ++which) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i * 13 + which * 17 + seed) * .017f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, input.data(), packed.data(), 0, 1021, d, nullptr);
        auto * tensor = which ? v : k;
        for (int h = 0; h < nkv; ++h) for (int64_t row = 0; row < n; ++row)
            std::memcpy(static_cast<char *>(tensor->data) + h * tensor->nb[2] + row * tensor->nb[1],
                        packed.data() + ((row + h * 23) % 1021) * tensor->nb[1], tensor->nb[1]);
    }
    if (mask) for (int column = 0; column < mask_columns; ++column) for (int64_t row = 0; row < n; ++row)
        static_cast<ggml_fp16_t *>(mask->data)[column * n + row] = ggml_fp32_to_fp16(
            column > 0 || mode == 2 || (mode == 1 && row >= n / 3) ? -INFINITY : 0);
    if (sinks) for (int h = 0; h < nq; ++h) static_cast<float *>(sinks->data)[h] = .05f * h;
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.f / std::sqrt(float(d)), 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto plan = ggml_graph_plan(graph, threads, nullptr);
    std::vector<uint8_t> workspace(plan.work_size);
    plan.work_data = workspace.data();
    auto run = [&](bool ref) {
        plan.use_ref = ref;
        auto start = std::chrono::steady_clock::now();
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) std::abort();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    run(true);
    std::vector<double> ref_samples;
    for (int i = 0; i < 3; ++i) ref_samples.push_back(run(true));
    std::sort(ref_samples.begin(), ref_samples.end());
    const double ref_seconds = ref_samples[1];
    std::vector<float> reference(nq * d);
    std::memcpy(reference.data(), out->data, reference.size() * sizeof(float));
    run(false); // warm both code paths, then report median of 3 actual dispatch calls
    std::vector<double> samples;
    for (int i = 0; i < 3; ++i) samples.push_back(run(false));
    std::sort(samples.begin(), samples.end());
    double max_abs = 0, sq_error = 0, sq_ref = 0;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float actual = static_cast<float *>(out->data)[i];
        finite &= std::isfinite(actual) && std::isfinite(reference[i]);
        const double delta = actual - reference[i];
        max_abs = std::max(max_abs, std::abs(delta));
        sq_error += delta * delta;
        sq_ref += double(reference[i]) * reference[i];
    }
    const double nmse = sq_error / std::max(sq_ref, 1e-30);
    const bool passed = finite && max_abs <= 5e-5 && nmse <= 1e-6;
    std::printf("{\"context\":%lld,\"threads\":%d,\"mode\":%d,\"seed\":%d,\"mask_columns\":%d,\"q_heads\":24,\"kv_heads\":4,\"head_dim\":256,\"kv_bytes\":%zu,\"reference_seconds\":%.9f,\"median_seconds\":%.9f,\"speedup_vs_reference\":%.6f,\"max_abs\":%.9g,\"nmse\":%.9g,\"finite\":%s,\"passed\":%s}\n",
                static_cast<long long>(n), threads, mode, seed, mask_columns, 2 * kv_bytes, ref_seconds, samples[1], ref_seconds / samples[1], max_abs, nmse, finite ? "true" : "false", passed ? "true" : "false");
    ggml_free(ctx);
    return passed && ref_seconds / samples[1] >= min_speedup ? 0 : 1;
}
