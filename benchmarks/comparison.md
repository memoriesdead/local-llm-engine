# LLM speed comparison

Checked 2026-10-30. **Our saved result is 310.84 tokens/s for the Claude Code API portion on one RTX 5090.** The full invocation was 146.52 tokens/s. These are different clocks.

This page covers **24 detailed local runtime/source projects, 25 additional source projects, four apps/frontends and five hosted providers (ten provider/model rows)**. It is a linked coverage list, not every fork and not a matched leaderboard. **Not measured here** means we did not run that comparison; it does not mean the project has no benchmarks.

## Our measured reference

| Measurement | Tokens/s | Clock |
| --- | ---: | --- |
| Claude Code SDK API portion | 310.84 | SDK API portion; excludes complete fresh invocation |
| Complete Claude Code invocation | 146.52 | Whole fresh invocation |
| Direct code API request | 371.80 | Request wall clock including API overhead |

Qwen-family 27B, one request at a time. Context was configured for 98,304 tokens; a separate retrieval check used 90,000 actual input tokens. The speeds above came from short code input, not a full-context test. These are historical whole-stack observations, not a measured gain from this patch bundle. The public source export has not had a fresh GPU run. [Reference data](reference.json).

## Local runtimes

The numbers below are **their publishers' results**, using their own models, settings and clocks. They are not reruns of our request. A larger number in this table does not establish a faster result on the same work.

| Runtime / primary source | What it is | Selected publisher result, tokens/s | Conditions and comparison limit |
| --- | --- | --- | --- |
| [vLLM](https://docs.vllm.ai/en/latest/) | General serving runtime | Not measured here | No matched run in this comparison. |
| [SGLang](https://github.com/sgl-project/sglang) | Language-model and multimodal serving | Not measured here | No matched run in this comparison. |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA inference runtimes and kernels | Not measured here | No matched run in this comparison. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ inference across hardware backends | Not measured here | No matched run in this comparison. |
| [NInfer upstream](https://github.com/Neroued/ninfer) | Specialized C++/CUDA engine | 147.7 single decode; 922.4 at eight requests | RTX 5090; Qwen3.8-27B NVFP4, FP8 row-256 KV, MTP3, CUDA graphs; 8,192 generated tokens per request; steady decode, not full-request wall time. |
| [NInfer-all / WaveCut fork](https://github.com/iamwavecut/ninfer-all/blob/master/docs/performance.md) | C++/CUDA fork used as this package's base | 233.5 short-chat decode | October 2026 publisher campaign: RTX 5090 600 W board, Qwen3.8 groupwise-int, rk8v4 KV, DFlash2 K7, build 796f98500 / CUDA 13.1; five 512-token short-chat answers, greedy, thinking off. |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 quantized consumer-GPU inference | Not measured here | No matched run in this comparison. |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | Compression, deployment and serving toolkit | Not measured here | No matched run in this comparison. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Compiled deployment engine | Not measured here | No matched run in this comparison. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple-silicon inference with MLX | Not measured here | Different hardware platform; no matched run here. |
| [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) | PaddlePaddle-based LLM/VLM deployment | Not measured here | No matched run in this comparison. |
| [Chitu](https://github.com/thu-pacman/chitu) | Language-model inference framework | Not measured here | No matched run in this comparison. |
| [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) | Text-generation server; repository says maintenance mode | Not measured here | No matched run in this comparison. |
| [LightLLM](https://github.com/ModelTC/LightLLM) | Python inference and serving framework | Not measured here | No matched run in this comparison. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Heterogeneous inference framework | Not measured here | No matched run in this comparison. |
| [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | Local model serving | Not measured here | The former SJTU-IPADS repository redirects here; no matched run. |
| [Aphrodite / Sonar](https://github.com/dphnAI/sonar) | Serving engine; former Aphrodite URL redirects here | Not measured here | No matched run in this comparison. |
| [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | Small inference-engine implementation | 1,434.13 aggregate | RTX 4070 Laptop 8 GB; Qwen3-0.6B; 256 total sequences, random 100–1,024-token inputs/outputs; 133,966 output tokens / 93.41 s. Request count is not stated concurrency. |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Rust inference runtime | Not measured here | No matched run in this comparison. |
| [DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII) | DeepSpeed-backed inference serving | Not measured here | No matched run in this comparison. |
| [ExLlamaV2](https://github.com/turboderp-org/exllamav2) | Consumer-GPU inference library | Not measured here | No matched run in this comparison. |
| [mkyt single-GPU engine](https://github.com/mkyt/qwen3-27b-5090/blob/main/docs/benchmarks.md) | Specialized Qwen3.8-27B C++/CUDA engine | 386 code / 258 prose decode | RTX 5090 at 400 W; gdnfp4 weights, NVFP4 KV, DFlash2; median of three 1,024-token runs. Rate excludes first-token wait; temperature 1, top-k 20, top-p 0.95, thinking off. |
| [Ignis](https://github.com/gpillon/ignis) | Rust/CUDA single-GPU engine | 367 coding geometric mean | RTX 5090, Qwen3.8-27B NVFP4, hq-e8-2b KV, DFlash2 K7, greedy, short inputs; one lane, eight coding tasks; runs configured for 262,144 context, not that many actual input tokens. |
| [Token Rush](https://github.com/zyhector/token-rush) | Specialized single-stream engine | 358 code / 229 essay decode | One RTX 5090; Qwen3.8-27B text only, GPTQ int4 group-128 weights (4.25 bits/weight), in-graph DFlash2; batch one, short context, greedy. Different target representation. |

The NInfer and NInfer-all rows identify separate source projects. QUASAR names an artifact/checkpoint lineage, not another independently counted engine. This package adds narrow NInfer and llama.cpp patches; it does not include or accelerate all projects listed here.

The mkyt number uses time from first output token to last. It excludes the initial wait that our direct API clock includes. Its headline sum of per-stream rates is also different from dividing all tokens by a common wall interval; no such sum is ranked against our single response.

## Apps and frontends

| Project / primary source | Why it is separate | Result |
| --- | --- | --- |
| [Ollama](https://docs.ollama.com/) | Application, model management and API; local and cloud routes exist. | Not measured here |
| [LM Studio](https://lmstudio.ai/docs/app/offline) | Desktop app/local server; downloads and selects separately packaged runtimes. | Not measured here |
| [Jan](https://github.com/janhq/jan) | Desktop model application; backend/model choice determines generation speed. | Not measured here |
| [Open WebUI](https://github.com/open-webui/open-webui) | Interface to inference backends; not a separate decode-kernel benchmark. | Not measured here |

## Hosted providers and frontier models

These services run on provider hardware. We sent no requests to them for this comparison. The model overviews identify the frontier families below; they do not give a matched output-speed measurement here. The open-weight models served by Cerebras and Groq are separate rows, not substitutes for the closed frontier models.

| Provider / primary source | Model(s) on checked page | Reported speed |
| --- | --- | --- |
| [OpenAI](https://developers.openai.com/api/docs/models) | GPT-6 Astra / GPT-6.1 Sol | Not measured here |
| [Anthropic](https://platform.claude.com/docs/en/models/overview) | Claude Fable 5.1 / Opus 5.5 / Sonnet 5.5 | Not measured here |
| [Google](https://ai.google.dev/gemini-api/docs/models) | Gemini 4 Argon / Gemini 3.8 Flash | Not measured here |
| [Cerebras](https://inference-docs.cerebras.ai/models/overview) | Qwen 3.8 27B | ~1,850 tokens/s, publisher catalog |
| [Cerebras](https://inference-docs.cerebras.ai/models/overview) | gpt-oss-120b | ~3,000 tokens/s, publisher catalog |
| [Groq](https://console.groq.com/docs/models) | qwen/qwen3.8-27b (preview; 131,072 context / 16,384 max output) | 450 tokens/s, publisher catalog (preview) |
| [Groq](https://console.groq.com/docs/models) | openai/gpt-oss-120b | 500 tokens/s, publisher catalog |
| [Groq](https://console.groq.com/docs/models) | openai/gpt-oss-20b | 1,000 tokens/s, publisher catalog |
| [Groq](https://console.groq.com/docs/models) | llama-3.3-70b-versatile | 280 tokens/s, publisher catalog |
| [Groq](https://console.groq.com/docs/models) | llama-3.1-8b-instant | 560 tokens/s, publisher catalog |

Cerebras and Groq numbers are provider catalog figures. The cited tables do not specify matching input/output lengths, request concurrency, sampling, chip count or a full-request/SDK clock. Keep those conditions unknown. They are not a one-5090 result or a promise of application latency. Tokens also differ across model tokenizers.

## How to read the numbers

- Decode rate measures generated output over a specified generation interval. It can exclude first-token wait.
- Full-request and complete SDK/application time include additional work.
- Prefill tokens/s measures processing input, not generating output.
- Aggregate tokens/s combines requests; total request count is not necessarily concurrent request count.
- Matching speed requires the same checkpoint/precision, hardware, input and output lengths, cache state, sampling, speculation and concurrency, followed by output checks. No such external matched run was performed here.

This page makes no fastest-engine or model-quality claim. Details, numeric conditions, unknown fields and checked source URLs are in [comparison.json](comparison.json). All external figures are publisher-reported and may change after the checked date.

## Additional source index: forks, recipes and components

These **25 additional projects** extend the link coverage. They include forks, recipes, decoding methods, kernel libraries and specialized source projects. Their public repository links/metadata were checked on 2026-10-30. They are **not 25 additional independently benchmarked engines**, and this index adds no speed, compatibility or quality result. Canonical GitHub repository URLs are deduplicated against the detailed rows and other existing sources.

| Public source | Type | Result |
| --- | --- | --- |
| [zhongkaifu/TensorSharp](https://github.com/zhongkaifu/TensorSharp) | Specialized runtime/source project | Not measured here |
| [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) | Specialized runtime/source project | Not measured here |
| [kacper-daftcode/qwentin](https://github.com/kacper-daftcode/qwentin) | Specialized runtime/source project | Not measured here |
| [kekzl/imp](https://github.com/kekzl/imp) | Specialized runtime/source project | Not measured here |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Kernel library | Not measured here |
| [z-lab/dflash](https://github.com/z-lab/dflash) | Speculative-decoding method | Not measured here |
| [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | Speculative-decoding method | Not measured here |
| [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | Speculative-decoding method | Not measured here |
| [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | Speculative-decoding method | Not measured here |
| [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) | Long-context component/method | Not measured here |
| [LukasParke/qwen38-27b-5090](https://github.com/LukasParke/qwen38-27b-5090) | Recipe/setup project | Not measured here |
| [thomgardiner/sglang-sm120-nvfp4](https://github.com/thomgardiner/sglang-sm120-nvfp4) | Backend patch | Not measured here |
| [flashrt-project/FlashRT](https://github.com/flashrt-project/FlashRT) | Specialized runtime/source project | Not measured here |
| [Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix) | Specialized runtime/source project | Not measured here |
| [Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco) | Backend fork | Not measured here |
| [adrienbrault/qwen3.8-27b-rtx5090](https://github.com/adrienbrault/qwen3.8-27b-rtx5090) | Multiple-GPU recipe | Not measured here |
| [splickz/ninfer-yarn-nvfp4](https://github.com/splickz/ninfer-yarn-nvfp4) | Long-context recipe | Not measured here |
| [varjoranta/turboquant-vllm](https://github.com/varjoranta/turboquant-vllm) | Long-context component/method | Not measured here |
| [BigBIueWhale/qwen_38_agent_setup](https://github.com/BigBIueWhale/qwen_38_agent_setup) | Long-context recipe | Not measured here |
| [tsarihan/qwen3.8-27b-rtx5090-playbook](https://github.com/tsarihan/qwen3.8-27b-rtx5090-playbook) | Recipe/setup project | Not measured here |
| [alphastorm/ninfer](https://github.com/alphastorm/ninfer) | Backend fork | Not measured here |
| [kurmetov/qwencore](https://github.com/kurmetov/qwencore) | Specialized runtime/source project | Not measured here |
| [r0b0tlab/qwen38-exl3-dflash2](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | Recipe/setup project | Not measured here |
| [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | Edge inference project | Not measured here |
| [Intelligent-Internet/Qwen3.8-Inference-MetaZenith](https://github.com/Intelligent-Internet/Qwen3.8-Inference-MetaZenith) | Backend patch | Not measured here |
