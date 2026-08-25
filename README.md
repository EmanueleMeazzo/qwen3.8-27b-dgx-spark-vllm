# Qwen3.8-27B on DGX Spark with vLLM — measured recipe

Optimized, experiment-backed configuration for serving **Qwen3.8-27B** (hybrid
GDN-linear-attention + full attention, native VLM, 262K context, in-checkpoint MTP head)
on an NVIDIA **DGX Spark** (GB10, sm_121, aarch64) using **vLLM**.

Every number below was measured on this machine (see `results/*.json`, `logs/`).
Benchmark method: net decode tok/s = server-reported `completion_tokens` / wall time after
first token, median of 3 streamed runs per probe, temp 0.7 / top_p 0.8, thinking off,
natively-sized probes (~700 tok code-gen, ~700 tok essay, ~300 tok chat).

---

## TL;DR — copy-paste config

```bash
docker pull nvcr.io/nvidia/vllm:26.07-py3

docker run -d --name vllm-qwen38 \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cpuset-cpus 5-9,15-19 \
  -p 127.0.0.1:8000:8000 \
  -v $HOME/dev/qwen3.8/hf-cache:/hf \
  -e HF_HOME=/hf \
  nvcr.io/nvidia/vllm:26.07-py3 \
  vllm serve unsloth/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen3.8-27b \
    --port 8000 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

First boot downloads ~22 GB of weights and takes ~10 min (torch.compile + graph capture);
subsequent boots ~5 min warm. Then:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role":"user","content":"Explain unified memory in two sentences."}]
}'
```

Measured performance of this exact config (this box):

| Metric | Value |
|---|---|
| Single-stream decode | **~28 tok/s code · ~19 tok/s essay · ~22 tok/s chat** |
| No-spec reference | 11.2 tok/s flat → MTP gives **~1.7–2.5×** |
| TTFT (short prompts, warm) | ~0.26 s |
| Concurrency (essay probe, aggregate) | 20 @1 · 35 @2 · 69 @4 · 119 @8 · **193 tok/s @16 streams**, 0 errors, TTFT ≤1.15 s |
| KV pool | 2.33M tokens (≈8.9× full-262K requests) |
| Quality gate | 12/12 exact-match sanity answers |

---

## 1. What this model is

`Qwen/Qwen3.8-27B` — arch `Qwen3_5ForConditionalGeneration` ("Qwen3.5" family):
- 64 layers: 48 GDN linear-attention + 16 full attention (`full_attention_interval: 4`)
- Vision tower included (serves images/video out of the box)
- Native 262,144-token context; extendable to 1M via YaRN (validated points: factor 2× @512K, 4× @1M)
- **In-checkpoint MTP draft layer** (`mtp_num_hidden_layers: 1`) — free speculative decoding, no extra download
- Thinking mode on by default; sampling defaults from `generation_config.json` (temp 1.0 / top_p 0.95 / top_k 20 thinking; 0.7 / 0.8 / 20 non-thinking)

## 2. Engine: container choice (measured)

| Container | Verdict |
|---|---|
| `nvcr.io/nvidia/vllm:25.12.post1-py3` (preinstalled here) | ❌ too old: no `Qwen3_5ForConditionalGeneration` |
| `vllm/vllm-openai:*` | arm64 images exist but NGC is the Spark-validated path |
| **`nvcr.io/nvidia/vllm:26.07-py3`** | ✅ vLLM 0.24.0-dev; has `Qwen3_5*`, `Qwen3_5MTP`, mtp/eagle/ngram/dflash spec methods |

Docker basics that matter on Spark: `--gpus all --ipc=host --ulimit memlock=-1
--ulimit stack=67108864 --shm-size 32g`.

## 3. Best quantization (measured)

| Checkpoint | Scheme | code | essay | chat | quality | Notes |
|---|---|---|---|---|---|---|
| **unsloth/Qwen3.8-27B-NVFP4** ✅ | mixed FP8+FP4 | **28.3** | **19.3** | **22.3** | 12/12 | smallest weight read, biggest KV pool |
| Inferact/Qwen3.8-27B-NVFP4 | uniform W4A4 modelopt | 21.2 | 14.3 | 17.6 | 12/12 | −25% vs winner |
| Qwen/Qwen3.8-27B-FP8 | block FP8 | 20.6 | 13.8 | 15.9 | died mid-run | needs `VLLM_USE_DEEP_GEMM=0` on GB10 (DeepGemm "Unknown recipe" assert); still slowest + unstable |

All three fit comfortably in 128 GB unified memory (NVFP4 ≈ 24.6 GiB checkpoint).
MXFP4 checkpoints do not work on NVIDIA hardware (missing linear-method support).

Why NVFP4 wins: GB10 decode is LPDDR5X-bandwidth-bound (no-spec ≈ 9.8–11.2 tok/s for every
quant — that's ~270 GB/s ÷ weight bytes). Less weight bytes per token = more tokens/s.

## 4. Best speculative decoding (measured)

The in-checkpoint **MTP head** is the right drafter on vLLM — zero extra setup.

`num_speculative_tokens` sweep on unsloth NVFP4:

| spec tokens | code | essay | chat | stability under concurrency |
|---|---|---|---|---|
| 0 | 11.2 | 11.2 | 11.3 | stable |
| 2 | 24.7 | 19.0 | 22.0 | not tested (dominated by 3) |
| **3** ✅ | 28.3 | 19.3 | 22.3 | **stable through 16 streams** (validated: FINAL-recommended) |
| 4 | 30.8 | 18.4 | 22.2 | ❌ CUDA illegal-memory-access at ≥4 streams |
| 6 (single-user code profile) | 33.8 | 17.3 | 19.9 | quality 12/12; same crash-risk class as 4 — **single-stream only** |

More draft tokens help predictable/code output and hurt prose (acceptance drops).

⚠️ **Known bug (NGC 26.07 build)**: with `num_speculative_tokens=4`, ≥4 concurrent requests
crash EngineCore with `CUDA error: an illegal memory access`. Single-stream is unaffected.
A per-batch schedule (`num_speculative_tokens_per_batch_size=[[1,1,4],[2,999,3]]`) is the
idea to watch once upstream fixes the kernel bug — a first attempt with wrong ranges
(`[1,2,4]`) reproduced the crash at batch=2. Until then: **flat 3 everywhere.**

Rejected alternatives:
- **TRITON_ATTN backend**: identical perf at bs=1 despite enabling FULL_AND_PIECEWISE graphs with spec-decode.
- **DFlash2 drafter** (`method:"dflash"`): NGC 26.07 predates vllm#52816's `DFlash2DraftModel` — unsupported.
- **ngram**: SGLang box measured it ~30% under MTP on this same model; skipped.

Sampling robustness: greedy vs temp 0.7 differ by <3% throughput; thinking mode
(temp 1.0) costs ~25% decode speed. Use thinking when quality demands it, not by default.

## 5. Serving parameters (measured)

| Flag | Value | Why |
|---|---|---|
| `--gpu-memory-utilization` | **0.90** | 0.85→KV 2.11M tok; 0.90→2.33M tok, no instability. Do NOT exceed ~0.92: OOM on unified memory can hard-wedge the box |
| `--max-num-seqs` | 16 | matches concurrency ladder; capping to 4 queues badly at 8+ streams (TTFT 20–42 s) |
| `--max-num-batched-tokens` | 8192 | good TTFT/throughput balance at these prompt sizes |
| `--kv-cache-dtype fp8` | yes | ~2× KV density; standard on this model |
| `--max-model-len` | 262144 | native. For 512K/1M add YaRN hf-overrides (factor 2/4, see model card) + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `--cpuset-cpus 5-9,15-19` | X925 cores only | big.LITTLE box; scheduler/tokenizer never land on A725 efficiency cores (SGLang box: +2–7% decode) |
| attention backend | default (FlashInfer) | TRITON_ATTN measured identical |
| prefix caching | n/a | engine keeps it off for this hybrid arch |

## 6. Operational notes

- **Client usage**: thinking mode is the template default; send `"chat_template_kwargs":
  {"enable_thinking": false}` for direct answers. Tool calling works via OpenAI-style
  `tools` + `tool_choice:"auto"` (parser already enabled). Sampling: temp 0.7/top_p 0.8
  (non-thinking) or 1.0/0.95 (thinking), top_k 20.
- **First long prefill after cold start** pays Triton/compile warmup; warm boots are cached
  in `/root/.cache/vllm` (mount it if you restart containers often — this repo mounts
  `hf-cache/vllm-cache` there).
- **Watch the box**: if the host ever wedges during experiments, it was memory — drop
  `--gpu-memory-utilization`.
- **Monitoring**: `curl localhost:8000/metrics` — `vllm:spec_decode_*` counters and
  `Avg Draft acceptance rate` in logs tell you the drafter is actually working.

## 7. Reproduce / re-tune everything in this repo

```bash
git clone https://github.com/EmanueleMeazzo/qwen3.8-27b-dgx-spark-vllm.git
cd qwen3.8-27b-dgx-spark-vllm

# full experiment: serve -> wait -> bench -> quality gate -> stop -> collect JSON
scripts/run_experiment.sh myrun "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90"

scripts/bench.py --port 8000 --probes code,essay,chat --ladder 1,2,4,8   # manual bench
scripts/quality.py --port 8000                                           # sanity gate
```

Weights and the vLLM compile cache land in `hf-cache/` (git-ignored, ~83 GB after a
full sweep). Every knob in `scripts/serve.sh` is an env var — see the header of that
file for the list.

### Repo layout

| Path | What |
|---|---|
| `scripts/serve.sh` | parametric container launcher (model, spec tokens, mem util, ctx, cpuset, YaRN) |
| `scripts/bench.py` | streamed decode benchmark; probes + concurrency ladder → JSON |
| `scripts/quality.py` | 12-question exact-match sanity gate |
| `scripts/run_experiment.sh` | one-command serve→bench→quality→stop cycle |
| `scripts/phase*.sh`, `final_validation.sh` | the exact sweeps that produced the tables above |
| `results/*.json` | every measurement in this README, raw |
| `logs/` | server logs from the final validated boot |
| `research/SOURCES.md` | third-party material consulted (snapshots themselves not committed) |

### Benchmark caveats

Run-to-run variance ±5%; treat deltas <15% on the code probe as noise (its token count
varies with output length). The essay probe is the tightest discriminator (±1% within a
boot). The box drifts slightly under sustained load (power cap) — compare within a session.

---

*Generated 2026-08-25 on gn100-6f38 (DGX Spark, driver 580.173.02, CUDA 13.3 userland /
580 kernel via forward-compat). Recipe scripts: `scripts/serve.sh`, `bench.py`,
`quality.py`, `run_experiment.sh`.*
