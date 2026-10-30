# How to compare local LLM speed

Use the same job for every engine. A different model, quantization or output
length is a different test.

## Record the setup

- GPU, CPU, RAM, driver and runtime versions.
- Engine version or commit, model source revision and weight checksum.
- Tokenizer, weight precision, KV-cache precision and context settings.
- Actual input/output token counts, sampler settings and concurrent requests.
- Whether the prompt cache is empty, warm, or reused across requests.
- The command and public test input, or an input checksum if it cannot be shared.

Keep model files unchanged during the test. Different file formats or conversion
recipes must be disclosed; the same model family is not proof of identical weights.

## Keep the clocks separate

| Metric | Calculation |
| --- | --- |
| Request throughput | Generated output tokens / complete API request wall seconds |
| Client throughput | Generated output tokens / complete client invocation wall seconds |
| Decode throughput | Generated tokens / the engine's stated decode interval |
| Time to first token | Request start to first nonempty output token |
| Prefill throughput | Input tokens / the stated prefill interval |
| Aggregate throughput | Total completed output tokens / one common wall interval |

Use the runtime's token counts, not a character-count estimate. Record the
token convention for decode timing: first-to-last-token timing usually excludes
the first token from its numerator. Do not relabel prefill or concurrent totals
as the speed of one response.

## Run and report

Check available resources first. Keep competing jobs intact. Use a small warmup,
then at least five measured runs of the same workload and report the median,
range, sample count and errors. Separate first-run startup from warmed inference.
Check that the output is valid; a fast error, truncated answer or failing code
sample is not a passing result.

The reference numbers in this repository are saved development observations,
with their original clocks and limits. They are not a fresh head-to-head test
of every runtime in the [comparison](comparison.md). Hosted figures stay labeled
as provider reports unless the same workload and measurement boundary are known.
Do not publish private prompts, credentials or local machine paths in results.
