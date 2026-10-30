<h1 align="center">Local LLM Engine</h1>

<p align="center">
  <strong>311 tokens/s through Claude Code on one RTX 5090.</strong><br>
  A small source patch package for NInfer and llama.cpp.
</p>

<p align="center">
  <a href="#measured-speed">Measured speed</a> ·
  <a href="#compare-the-clocks">Comparisons</a> ·
  <a href="#use">Quick start</a> ·
  <a href="benchmarks/protocol.md">Benchmark method</a>
</p>

![Saved local LLM benchmark: 310.84 tokens/s for the Claude Code SDK API portion, 146.52 for a full invocation and 371.80 for a direct API request. Context was configured for 98,304 tokens; a separate retrieval check used 90,000 actual input tokens. Historical short-prompt whole-stack results; no attributable patch gain established and no fresh public-export GPU benchmark.](assets/benchmark-dashboard.svg)

## Measured speed

The saved development stack ran a **Qwen-family 27B model on one RTX 5090**, one
request at a time. Each speed below measures a different part of the work.

| Measurement | Result | Scope |
| --- | ---: | --- |
| Claude Code SDK API portion | **310.84 tokens/s** | API portion; excludes the complete fresh invocation |
| Full Claude Code invocation | **146.52 tokens/s** | Whole fresh invocation |
| Direct code API request | **371.80 tokens/s** | Request wall time including API overhead |
| Configured context | **98,304 tokens** | Configuration, not input length for the speed runs |
| Separate retrieval check | **90,000 input tokens** | Actual input tokens in a separate check |

These are **historical whole-stack results from short code prompts**. They do not
measure a speedup caused by this patch bundle, and the public source export has
not had a fresh GPU benchmark. Short-prompt speed does not establish full-context
speed. See the [saved reference data](benchmarks/reference.json) and
[measurement protocol](benchmarks/protocol.md).

## Compare the clocks

The [full comparison](benchmarks/comparison.md) links **24 detailed local runtime
projects, 25 additional source projects, four apps/frontends and five hosted
providers**. It is a source catalog, not a matched leaderboard or a benchmark of
every fork. The selections below retain their publishers' timing boundaries;
none was rerun against our request. Source pages were checked on **2026-10-30**.

### Selected local publisher reports

| Primary source | Publisher tokens/s | Model and hardware | Work and clock |
| --- | ---: | --- | --- |
| [NInfer upstream](https://github.com/Neroued/ninfer) | 147.7 single; 922.4 aggregate | Qwen3.8-27B NVFP4 · RTX 5090 | Steady decode; 8,192 output tokens/request; one or eight concurrent requests; MTP3 |
| [NInfer-all / WaveCut](https://github.com/iamwavecut/ninfer-all/blob/master/docs/performance.md) | 233.5 | Qwen3.8-27B groupwise-int · RTX 5090, 600 W board | Short-chat decode; five 512-token answers; greedy; DFlash2 K7 |
| [mkyt](https://github.com/mkyt/qwen3-27b-5090/blob/main/docs/benchmarks.md) | 386 code; 258 prose | Qwen3.8-27B gdnfp4 · RTX 5090 at 400 W | Median of three 1,024-token runs; first-to-last output token, excluding first-token wait; DFlash2 |
| [Ignis](https://github.com/gpillon/ignis) | 367 coding | Qwen3.8-27B NVFP4 · RTX 5090 | Decode geometric mean; eight short-input tasks, one lane, greedy; DFlash2 K7 |
| [Token Rush](https://github.com/zyhector/token-rush) | 358 code; 229 essay | Qwen3.8-27B GPTQ int4 · RTX 5090 | Single-stream decode; short context, greedy; in-graph DFlash2 |

Different checkpoints, quantization, prompts, sampling, speculation and clocks
prevent a speed ranking. In particular, decode can exclude the initial wait
included in our direct API clock, and aggregate throughput combines requests.
The [complete local table](benchmarks/comparison.md#local-runtimes) includes
vLLM, SGLang, TensorRT-LLM, llama.cpp, ExLlama, MLX-LM and other runtimes, with
full numeric conditions in [comparison.json](benchmarks/comparison.json).

### Hosted providers and frontier models

| Official provider source | Models in the checked catalog | Speed and hardware scope |
| --- | --- | --- |
| [OpenAI](https://developers.openai.com/api/docs/models) | GPT-6 Astra / GPT-6.1 Sol | Not measured here; provider hardware |
| [Anthropic](https://platform.claude.com/docs/en/models/overview) | Claude Fable 5.1 / Opus 5.5 / Sonnet 5.5 | Not measured here; provider hardware |
| [Google](https://ai.google.dev/gemini-api/docs/models) | Gemini 4 Argon / Gemini 3.8 Flash | Not measured here; provider hardware |
| [Cerebras](https://inference-docs.cerebras.ai/models/overview) | Qwen 3.8 27B | ~1,850 tokens/s, publisher catalog; provider hardware |
| [Groq](https://console.groq.com/docs/models) | qwen/qwen3.8-27b, preview | 450 tokens/s, publisher catalog; provider hardware |

We sent no hosted requests for this comparison. Cerebras and Groq serve
open-weight models in the selected rows; these are separate from the closed
frontier families above. Their catalog rates do not specify matching token
lengths, concurrency, sampling, chip count or API/SDK timing. They do not
establish one-5090 speed or application latency. Model quality was not compared,
and tokenizers differ. [All ten hosted model rows](benchmarks/comparison.md#hosted-providers-and-frontier-models)
and [all 25 additional source projects](benchmarks/comparison.md#additional-source-index-forks-recipes-and-components)
remain available in the full catalog.

## Use

Install Python 3.10+ and Git. Use a dedicated backend checkout at the exact
revision in its `backend.json`, then run:

```shell
python tools/engine.py list
python tools/engine.py check ninfer --checkout /path/to/ninfer
python tools/engine.py apply ninfer --checkout /path/to/ninfer
```

Use `llama-cpp` in place of `ninfer` for that backend. The installer checks the
revision, source targets and patch hashes, and refuses dirty tracked source.
It does not download a runtime, load a model or start inference.

Build instructions and numerical checks:

- [NInfer](backends/ninfer/README.md)
- [llama.cpp CPU patch](backends/llama-cpp/README.md)
- [Measurement details](benchmarks/reference.json)

Run the package tests with `python -B -m unittest discover -s tests`.
Check hardware capacity before building or running a backend.

## What is patched

The package contains two engine patches. Model weight files stay unchanged.

| Backend | Patch |
| --- | --- |
| NInfer | Exact-bit BF16 embedding gather for width 5120, aligned buffers and batches 32–128; existing fallback otherwise |
| llama.cpp | Experimental CPU q4 attention reuse for one-token decode with 24 query heads, four KV heads and dimension 256; existing fallback otherwise |

Model support comes from the backend. The patches do not add support for every
model or guarantee a speedup. The CPU patch remains unpromoted; an attributable
whole-model gain has not been established. No models, prompts, generated output
or hosted inference are included.

## Licenses

The original tools use [Apache-2.0](LICENSE). Backend code retains its
[Apache/MIT licenses and notices](LICENSES.md).
