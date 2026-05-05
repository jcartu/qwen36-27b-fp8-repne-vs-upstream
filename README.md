# Qwen3.6-27B FP8+MTP=3 — Repne fork vs Upstream vLLM v0.20.1

Head-to-head benchmark of the **same model and same config** on two different vLLM builds, on dual NVIDIA RTX PRO 6000 Blackwell (TP=2).

## TL;DR — Repne fork wins

| cell | Repne fork | Upstream v0.20.1 | Δ |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | 112.3 | **+7.0%** |
| c=1 ctx=131k | 93.7 | **99.0** | **−5.3%** |
| c=2 ctx=0 | **223.8** | 197.0 | **+13.6%** |
| c=2 ctx=131k | 183.4 | **183.9** | **−0.3%** |
| c=4 ctx=0 | **449.5** | 413.8 | **+8.6%** |
| c=4 ctx=131k | **347.4** | 345.0 | **+0.7%** |

Repne fork is meaningfully faster at short context multi-stream (c=2/c=4 at ctx=0), where most agent traffic actually sits. Long context is essentially a wash — both engines hit the same KV-bound ceiling at 131K.

**Recommendation:** Stay on the Repne fork. Upstream v0.20.1 is a clean fallback if Repne ever stops shipping, with negligible long-context cost and ~5–14% short-context regression.

## Setup (identical for both)

| | |
|---|---|
| Model | `Qwen/Qwen3.6-27B-FP8` |
| TP | 2 |
| max-model-len | 262144 |
| max-num-seqs | 128 |
| max-num-batched-tokens | 32758 |
| max-cudagraph-capture-size | 256 |
| GPU mem util | 0.85 |
| Speculative decoding | MTP, num_speculative_tokens=3 |
| Attention backend | flashinfer |
| Prefix caching | on |
| Reasoning parser | qwen3 |
| Tool parser | qwen3_coder |

## Differences

| | Repne fork | Upstream v0.20.1 |
|---|---|---|
| Image | `repne/vllm:latest` (`5e7583ca4df9`, May 5 2026) | `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` (`7ba11e462b5a`) |
| Engine version | `v0.1.dev16359+ga3e24c99b.d20260505` | `v0.20.1` |
| KV cache size | 1,846,472 tokens (7.04× max conc at 256K) | 1,828,129 tokens (6.97× max conc at 256K) |
| `--load-format instanttensor` | yes | dropped (Repne-only) |
| `draft_sample_method=gumbel` | yes | dropped (Repne-only — `pydantic.ValidationError: Unexpected keyword argument`) |
| Boot time | 140s (warm cache) | 342s (cold compile) |

## Methodology

- N=1 single-run per cell, 30s sustained-decode + 10s warmup
- `--skip-prefill` (decode-only measurement)
- `llm_decode_bench.py v0.4.8`
- Run order: c=1 ctx=0 → c=1 ctx=131k → c=2 ctx=0 → c=2 ctx=131k → c=4 ctx=0 → c=4 ctx=131k

## Detailed results

### TTFT and ITL (latency)

| cell | TTFT avg (ms) Repne / Upstream | TTFT p99 (ms) Repne / Upstream | ITL avg (ms) Repne / Upstream |
|---|---|---|---|
| c=1 ctx=0 | 65 / 69 | 69 / 73 | 8.19 / 8.85 |
| c=1 ctx=131k | 783 / 767 | 791 / 767 | 10.17 / 9.77 |
| c=2 ctx=0 | 242 / 395 | 838 / 1302 | 8.15 / 9.82 |
| c=2 ctx=131k | 1102 / 1104 | 1413 / 1413 | 10.49 / 10.36 |
| c=4 ctx=0 | 127 / 156 | 199 / 378 | 8.55 / 9.56 |
| c=4 ctx=131k | 1912 / 2050 | 3311 / 3251 | 11.02 / 11.06 |

### Spec-decode acceptance

| cell | Repne | Upstream |
|---|---|---|
| c=1 ctx=0 | 41.5% | 51.6% |
| c=1 ctx=131k | 69.4% | 44.7% |
| c=2 ctx=0 | 64.7% | 48.8% |
| c=2 ctx=131k | 56.0% | 43.7% |
| c=4 ctx=0 | 57.6% | 49.4% |
| c=4 ctx=131k | 54.3% | 59.5% |

Acceptance varies wildly (43-69%) at N=1 — these single-shot numbers are not reliable variance estimates. With N=3 the picture would tighten. The pattern suggests Repne's gumbel sampler has higher acceptance at multi-stream (where it matters), upstream's default is more uniform but lower in the high-throughput cells.

## Caveats

- **N=1, no variance bands.** The `c=4 ctx=0 = 449.5 tok/s` reading is +27% above this morning's EXP-1 baseline of 352.8. Could be hot tail, or could mean the morning regression was transient state we now don't reproduce. Don't draw firm conclusions from individual cells without N≥3.
- **Two Repne-only flags dropped from upstream** (`instanttensor` load format and `gumbel` draft sampler). This may understate upstream's relative perf if those flags are net-zero or negative on Repne.
- Upstream's first boot took 342s vs Repne's 140s due to torch.compile cold-cache. Subsequent boots would be similar.

## Files

- `repne-fork/c{1,2,4}_ctx{0,131072}.json` — raw bench tool output
- `repne-fork/c{1,2,4}_ctx{0,131072}.log` — full bench tool stdout
- `upstream-v0.20.1/` — same layout for upstream

## Related

- Earlier today: NVFP4-MTP experiment → https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment
- Earlier today: Repne dflash newimage validation → https://github.com/jcartu/repne-dflash-newimage
