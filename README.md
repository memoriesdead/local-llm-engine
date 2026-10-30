# Local LLM Engine

**311 tokens/s through Claude Code on one RTX 5090**, measured with a Qwen 27B
model and a short code prompt.

| Measurement | Result |
| --- | ---: |
| Claude Code SDK API portion | 310.84 tokens/s |
| Full Claude Code invocation | 146.52 tokens/s |
| Direct API code request | 371.80 tokens/s |
| Configured context | 98,304 tokens |
| Separate retrieval check | 90,000 actual input tokens |

These are saved whole-stack results. This public source package has not had a
fresh GPU test, and those numbers do not measure the speedup from its patches.
The short-prompt speeds were not measured at the full configured context.

The package contains two engine patches. Model weight files stay unchanged.

| Backend | Patch |
| --- | --- |
| NInfer | Exact-bit BF16 embedding gather for width 5120, aligned buffers and batches 32–128; existing fallback otherwise |
| llama.cpp | Experimental CPU q4 attention reuse for one-token decode with 24 query heads, four KV heads and dimension 256; existing fallback otherwise |

Model support comes from the backend. The patches do not add support for every
model or guarantee a speedup. The CPU patch remains unpromoted; an attributable
whole-model gain has not been established. No models, prompts, generated output
or hosted inference are included.

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

The original tools use [Apache-2.0](LICENSE). Backend code retains its
[Apache/MIT licenses and notices](LICENSES.md).
