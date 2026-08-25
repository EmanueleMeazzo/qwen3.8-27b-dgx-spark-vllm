# Qwen3.8-27B on DGX Spark with vLLM — measured recipe

Optimized, experiment-backed configuration for serving **Qwen3.8-27B** (hybrid
GDN-linear-attention + full attention, native VLM, 262K context, in-checkpoint MTP head)
on an NVIDIA **DGX Spark** (GB10, sm_121, aarch64) using **vLLM**.

Every number below was measured on this machine (see `results/*.json`, `logs/`). Tables are
generated from those files by `scripts/summarize.py`, not typed by hand.

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
subsequent boots ~5 min warm.

Measured performance of this exact config (this box):

| Metric | Value |
|---|---|
| Single-stream decode | **~28 tok/s code · ~19 tok/s essay · ~22 tok/s chat** |
| No-spec reference | 11.2 tok/s flat → MTP gives **~1.7–2.5×** |
| TTFT (short prompts, warm) | ~0.27 s |
| Concurrency (essay probe, aggregate) | 20 @1 · 35 @2 · 69 @4 · 119 @8 · **193 tok/s @16 streams**, 0 errors |
| KV pool | 2.33M tokens (≈8.9× full-262K requests) |
| Quality vs unquantized BF16 | 95.7% identical greedy choices; identical capability score |

**Two documented alternatives** (§3): `RadixArk/Qwen3.8-27B-NVFP4` for ~+11% decode at a
measurably larger drift from the unquantized model, and `lued/Qwen3.8-27B-INT8-W8A16-MTP` when
you need BF16-identical output — it reproduces BF16's greedy choices exactly, at 1.53× BF16's
speed.

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

**26.07 is the newest tag NVIDIA publishes** — verified against the NGC registry tag list
(`nvcr.io/v2/nvidia/vllm/tags/list`; nothing above 26.07 exists). Every "just upgrade the
container" idea below is therefore closed until NVIDIA ships a newer image.

Docker basics that matter on Spark: `--gpus all --ipc=host --ulimit memlock=-1
--ulimit stack=67108864 --shm-size 32g`.

## 3. Quantization (measured)

Candidates were filtered **before** downloading: every checkpoint here retains the
in-checkpoint MTP head (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors, confirmed from each
repo's `config.json` + `model.safetensors.index.json`). A quant that strips it loses ~2×
decode and is not comparable at all.

| Checkpoint | Scheme | Weights | code | essay | chat | TTFT | MTP accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4 (mixed FP8+FP4) | 21.8 GiB | 28.0 | 19.2 | 22.2 | 0.29 s | 0.600 |
| `RadixArk/Qwen3.8-27B-NVFP4` | NVFP4 (MTP unquantized) | 20.4 GiB | 30.6 | 21.3 | 24.8 | 0.27 s | 0.593 |
| `RedHatAI/Qwen3.8-27B-INT4` | INT4 W4A16 | 18.1 GiB | 25.2 | 17.2 | 20.4 | 0.28 s | 0.605 |
| `philbert440/Qwen3.8-27B-W4A16-AWQ` | INT4 W4A16 AWQ | 18.2 GiB | 25.7 | 17.8 | 21.8 | 0.28 s | 0.606 |
| `lued/Qwen3.8-27B-INT8-W8A16-MTP` | INT8 W8A16 | 29.4 GiB | 18.9 | 12.7 | 14.8 | 0.38 s | 0.598 |
| `Qwen/Qwen3.8-27B` | BF16 (unquantized) | 51.7 GiB | 11.9 | 8.3 | 10.1 | 0.60 s | 0.605 |

| Checkpoint | PPL | top-1 agree vs ref | KL(ref‖x) | reasoning | code |
|---|---:|---:|---:|---:|---:|
| `unsloth/Qwen3.8-27B-NVFP4` | 2.880 | 0.957 | 0.0198 | 14/24 | 8/8 |
| `RadixArk/Qwen3.8-27B-NVFP4` | 2.715 | 0.928 | 0.0256 | 12/24 | 8/8 |
| `RedHatAI/Qwen3.8-27B-INT4` | 2.809 | 0.971 | 0.0209 | 12/24 | 8/8 |
| `philbert440/Qwen3.8-27B-W4A16-AWQ` | 2.871 | 0.957 | 0.0184 | 15/24 | 8/8 |
| `lued/Qwen3.8-27B-INT8-W8A16-MTP` | 2.634 | 1.000 | 0.0010 | 14/24 | 8/8 |
| `Qwen/Qwen3.8-27B` | 2.639 | *reference* | *0* | 14/24 | 8/8 |

**How the quality columns work** — see §8. `top-1 agree` is the fraction of probed positions
where this checkpoint's greedy choice equals the unquantized BF16 model's; `KL(ref‖x)` is the
divergence of its next-token distribution from BF16's.

⚠️ **Never compare `PPL` across model runners.** Probing the *same checkpoint* three times
gives 2.8798 and 2.8777 on the V1 runner — repeatable to **0.073%** — but 3.0758 on the V2
runner, a **6.8%** shift. That is systematic, not noise: the V2 runner's sampling path yields
a measurably different likelihood for identical weights. Every checkpoint in the table above
was probed on V1, so the column is internally comparable; a V2 number dropped into it would
not be. `top-1 agree` and `KL` are far less sensitive to this (0.9565 on both V1 runs *and*
the V2 run), which is why the ranking claims below lean on them.

### What the numbers say

**Fewer weight bytes does not mean more tokens/s.** This is the headline correction to the
previous revision of this recipe, which reasoned that decode is LPDDR5X-bandwidth-bound and
concluded "less weight bytes per token = more tokens/s". Bandwidth-bound is right; the
inference from it is not. `RedHatAI/Qwen3.8-27B-INT4` is **17% smaller** than the recommended
NVFP4 checkpoint and decodes **~10% slower**. MTP acceptance is identical (0.605 vs 0.600), so
the drafter is not the cause: the W4A16 INT4 path dequantizes to bf16 before the GEMM and
simply does not convert its byte advantage into bandwidth. NVFP4 runs on GB10's native FP4
tensor cores.

The rule that *does* hold is bytes-within-a-kernel-family. Taking the recommended
checkpoint as the baseline and predicting every other checkpoint's rate as
`baseline_rate × baseline_bytes / bytes`:

| Checkpoint | Weights | essay tok/s | predicted from bytes | achieved | GEMM path |
|---|---:|---:|---:|---:|---|
| unsloth NVFP4 | 21.8 GiB | 19.17 | 19.17 | **1.00×** | native FP4 |
| RadixArk NVFP4 | 20.4 GiB | 21.33 | 20.49 | **1.04×** | native FP4 |
| BF16 | 51.7 GiB | 8.33 | 8.08 | **1.03×** | native bf16 |
| RedHatAI INT4 | 18.1 GiB | 17.24 | 23.09 | **0.75×** | W4A16 dequant |
| philbert AWQ | 18.2 GiB | 17.81 | 22.96 | **0.78×** | W4A16 dequant |
| lued INT8 | 29.4 GiB | 12.72 | 14.21 | **0.89×** | W8A16 dequant |

Every checkpoint on a **native** path — NVFP4 on GB10's FP4 tensor cores, BF16 on bf16 —
lands within 4% of what its byte count predicts. Every checkpoint that **dequantizes to bf16
before the GEMM** falls short: W8A16 by 11%, and *both* independent W4A16 checkpoints by
22–25%. Two unrelated W4A16 quantizations (different producers, different calibration)
landing at 0.75× and 0.78× is what makes this a property of the kernel path rather than of
any one checkpoint. It is also why the two smallest checkpoints in the table are not the
fastest.

Regenerate with `scripts/bandwidth_efficiency.py`.

**Bigger checkpoints run fine — and BF16 specifically is dominated by INT8.** BF16 loads in
51.9 GiB and still leaves a 1.53M-token KV pool (5.8× a full 262K-token request); it is
entirely usable at ~8–12 tok/s. It also scores *identically* to NVFP4 on the capability gate
(22/32), so there is no quality argument for paying 2.3× the decode cost.

More usefully: `lued/Qwen3.8-27B-INT8-W8A16-MTP` reproduces BF16 almost exactly — **100.0%
identical greedy choices and KL 0.00102, an order of magnitude below every 4-bit checkpoint
and far outside the measurement noise** — at 1.53× the decode
rate and 57% of the weight memory, and it matches BF16's capability score exactly (22/32).
If you have a reason to want reference-model behaviour
(evaluating a quant, reproducing a result, anything where "same output as the unquantized
model" is the requirement), INT8 is that, and plain BF16 is strictly the worse way to get it.

That also puts the 4-bit numbers in scale: the entire 4-bit tier sits at 0.93–0.97 greedy
agreement, where 8-bit sits at 1.00. Whether ~5% of greedy choices differing matters is a
product question, not a benchmark one — the capability gate cannot resolve it (§8).

**There is a faster NVFP4, and it is a trade.** `RadixArk/Qwen3.8-27B-NVFP4` decodes ~11%
faster than the incumbent on every probe with better TTFT. Two contributions: 6% fewer weight
bytes, and — more importantly — it is *uniform* NVFP4 with only the MTP head excluded from
quantization, where `unsloth/…-NVFP4` is a **mixed FP8+FP4** checkpoint. More layers on the
native FP4 path means both fewer bytes and faster GEMMs, which is why the gain exceeds the 6%
byte reduction. (Its MTP acceptance is 0.593, marginally *below* unsloth's 0.600, so the
drafter is not the source.)

The cost is fidelity: RadixArk diverges most from BF16 of all six checkpoints — 92.8%
greedy agreement vs 95.7% for both unsloth and philbert, and KL 0.0256 vs 0.0198/0.0184. Those gaps are real: the same-
checkpoint repeat runs put the noise on agreement at ~0.000 and on KL at ~3% relative, so a
2.9-point agreement gap (95.7% → 92.8%) and a 29% KL gap are both well outside it. Its capability score is 2
items lower (20/32 vs 22/32), which on its own is *not* significant — 32 binomial trials carry
±2.8 — but it points the same way as the fidelity metrics rather than against them.

RadixArk also posts the **lowest** perplexity of any quant (2.715 vs unsloth's 2.880), and
since PPL is repeatable to 0.073% within a runner that is a real effect rather than noise. It
is simply a different question: perplexity asks how well the checkpoint predicts the probe
text, and a quant can be sharper on that text while sitting further from BF16's distribution.
When the two disagree, the one that answers "is this still the same model" is KL.

**Recommendation.** There is no single winner; the checkpoints form a clean speed/fidelity
frontier, and which end you want is a product decision:

| want | pick | essay tok/s | KL vs BF16 |
|---|---|---:|---:|
| maximum throughput | `RadixArk/…-NVFP4` | **21.3** | 0.0256 (worst) |
| balanced default | **`unsloth/…-NVFP4`** | 19.2 | 0.0198 |
| best 4-bit fidelity | `philbert440/…-W4A16-AWQ` | 17.8 | **0.0184** |
| BF16-identical output | `lued/…-INT8-W8A16-MTP` | 12.7 | **0.0010** |

`unsloth/Qwen3.8-27B-NVFP4` stays the recommended default because it is the best
speed-per-unit-fidelity point on that frontier: 8% faster than the most faithful 4-bit
checkpoint for 8% more KL, and 10% slower than the fastest for 23% less. Nothing in the data
forces that choice — it is a judgement about where the knee is, and the table above is there
so you can pick a different one.

Note that `philbert440`, not unsloth, has the lowest divergence of any 4-bit checkpoint
(0.0184), and both tie on greedy agreement at 0.957. An earlier draft of this recipe claimed
unsloth was the most faithful quant; that was true only of the checkpoints measured at the
time.

MXFP4 checkpoints do not work on NVIDIA hardware (missing linear-method support).
`Qwen/Qwen3.8-27B-FP8` additionally needs `VLLM_USE_DEEP_GEMM=0` on GB10 (DeepGemm "Unknown
recipe" assert) and was both slowest and least stable of the schemes tried.

## 4. Speculative decoding (measured)

The in-checkpoint **MTP head** is the right drafter on vLLM — zero extra setup. Acceptance
rate is essentially constant across every checkpoint tested (0.593–0.605, BF16 included), so
the drafter is never what distinguishes them.

`num_speculative_tokens` sweep on unsloth NVFP4 (V1 runner unless noted):

| spec tokens | code | essay | chat | stability under concurrency |
|---|---|---|---|---|
| 0 | 11.2 | 11.2 | 11.3 | stable |
| 2 | 24.7 | 19.0 | 22.0 | not tested (dominated by 3) |
| **3** ✅ | 28.0 | 19.2 | 22.2 | **stable through 16 streams** |
| 4 | 30.8 | 18.4 | 22.2 | ❌ CUDA illegal-memory-access at ≥2 streams (`results/p5b-*`) |
| 6 (single-user code profile) | 33.8 | 17.3 | 19.9 | same crash class as 4 — **single-stream only** |

More draft tokens help predictable/code output and hurt prose (acceptance drops: 0.50 at
4 tokens vs 0.60 at 3).

### The MTP=4 crash is a V1-model-runner bug

The previous revision recorded the `num_speculative_tokens=4` illegal-memory-access as a
property of the NGC 26.07 build and recommended flat 3 everywhere. It is narrower than that:
it is specific to the **V1** model runner, which is what this model selects by default.

Re-running on the V2 runner (`-e VLLM_USE_V2_MODEL_RUNNER=1`), with a same-session
V2+MTP=3 control so the runner and the draft depth are not confounded:

| Config | 1 stream | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| V1 + MTP=4 (`p5b`) | 18.5 | **2 err** | **4 err** | **8 err** |
| V1 + MTP=3 (default) | 19.5 | 35.5 | 69.3 | 119.1 |
| V2 + MTP=3 (control) | 18.7 | 34.0 | 68.5 | 120.5 |
| V2 + MTP=4 | 18.3 | 33.4 | 64.5 | 97.2 |

Two separate conclusions, and the control is what separates them:

1. **The V2 runner is free.** At MTP=3 it matches V1 everywhere — 120.5 vs 119.1 tok/s at 8
   streams, 28.4/19.5/22.9 vs 28.0/19.2/22.2 single-stream. It costs nothing to enable.
   All four rows are same-session, so this is not a cross-boot artifact — and the V1 ladder
   re-run reproduced the earlier `FINAL-recommended` boot to within 0.1% (119.1 vs 119.2 at 8
   streams), which is also the cleanest available check that the box had not drifted across
   the sweep.
2. **MTP=4 is what costs throughput under load**, not the runner: 97.2 vs 120.5 tok/s at 8
   streams on the *same* runner. It buys ~11% single-stream code generation (31.1 tok/s) for
   ~19% aggregate throughput at 8 streams.

So the crash was never a reason to avoid MTP=4 — the V1 runner was. Flat **MTP=3 remains the
recommended default** because it is faster wherever more than one request is in flight. If
your workload is single-stream code generation, `VLLM_USE_V2_MODEL_RUNNER=1` plus
`num_speculative_tokens: 4` is now a safe configuration rather than a crashing one.


### DFlash2: backported, does not work on 26.07

`incoai/Qwen3.8-27B-DFlash2` is a block-diffusion drafter claiming better acceptance than MTP.
The previous revision recorded it as "NGC 26.07 predates vllm#52816 — unsupported". That is
half right, and the useful half is different: **the image already ships DFlash v1**; what it
lacks is DFlash2's grouped convolutions and candidate selector. Upstream added those on
2026-08-21, days after the image was cut, in **pure Python — no CUDA kernels**.

`dflash2-backport/` applies that change to the installed package (builds in seconds). It gets
further than expected — the registry resolves `DFlash2DraftModel`, target and draft both load,
the candidate selector compiles, graph capture completes — and then faults in `propose()` with
an illegal memory access.

The cause is a buffer-convention mismatch, not a missing feature: the container's DFlash v1
speculator pads `sample_idx_mapping` with `0` and stores `sample_pos` un-incremented, while
DFlash2's Triton kernels assume upstream's `-1` padding sentinel (`valid = req_state >= 0`
*is* the padding test) and a pre-incremented `sample_pos`. Both are adaptable, and both were
deliberately left alone: speculative decoding is supposed to be lossless, a guessed-at buffer
convention would produce silently mis-sampled drafts, and nothing downstream would flag it.
See `dflash2-backport/README.md`. Re-run it when NVIDIA ships an image built after 2026-08-21.

Rejected alternatives:
- **TRITON_ATTN backend**: identical perf at bs=1 despite enabling FULL_AND_PIECEWISE graphs.
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
  in `/root/.cache/vllm` (this repo mounts `hf-cache/vllm-cache` there). A *new* quantization
  scheme or model runner is a cold compile even with the cache mounted — budget ~10 min.
- **Watch the box**: if the host ever wedges during experiments, it was memory — drop
  `--gpu-memory-utilization`.
- **Monitoring**: `curl localhost:8000/metrics` — `vllm:spec_decode_*` counters and
  `Avg Draft acceptance rate` in logs tell you the drafter is actually working.
- ⚠️ **`echo` / prompt-logprobs return garbage on this architecture.** Asking the completions
  endpoint to score a prompt (`echo: true` + `logprobs`) produces wrong values for every
  non-final position: a token the normal decode path scores at −1.04 comes back as −17.13,
  with unrelated junk tokens as the claimed top-1. This is silent — the response is
  well-formed — so any perplexity or logprob-based evaluation built on `echo` will be
  confidently wrong. `scripts/fidelity.py` routes around it (§8). Suspected cause is the
  hybrid GDN layers, whose recurrent state is only correct at the final position.

## 7. Reproduce / re-tune everything in this repo

```bash
git clone https://github.com/EmanueleMeazzo/qwen3.8-27b-dgx-spark-vllm.git
cd qwen3.8-27b-dgx-spark-vllm

# full experiment: serve -> wait -> bench -> quality -> fidelity -> capability -> stop
scripts/run_experiment.sh myrun "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90"

# the quantization sweep in this README
scripts/fetch_models.sh RadixArk/Qwen3.8-27B-NVFP4 RedHatAI/Qwen3.8-27B-INT4
scripts/quant_sweep.sh "Q-radixark-nvfp4=RadixArk/Qwen3.8-27B-NVFP4"

# regenerate every table above from results/*.json
scripts/summarize.py --ref Q-bf16
```

Weights and the vLLM compile cache land in `hf-cache/` (git-ignored, ~180 GB after the full
sweep). Every knob in `scripts/serve.sh` is an env var — see the header of that file.

## 8. How quality is measured

The original `quality.py` — twelve one-hop questions like "what is 17×23" — scored **12/12 for
every checkpoint that booted at all**. It is a breakage detector, not a ranking, and it is
kept only as a fast boot gate. Ranking quants needs two things it cannot do:

**`scripts/fidelity.py` — distributional fidelity.** Probes ~69 fixed positions across four
passages (English prose, Python, worked mathematics, Chinese) and reports, per position, the
checkpoint's next-token distribution. Against a reference checkpoint it yields `top-1 agree`
(how often greedy decoding would pick the same token as unquantized BF16) and `KL(ref‖x)`.
This is the sensitive instrument: 69 positions, deterministic, no sampling.

It deliberately does **not** use `echo`/prompt-logprobs, which is broken here (§6). Every
probe instead feeds the true prefix and reads one next-token distribution through the normal
decode path. `--compare` scores saved probe files offline, so each checkpoint is measured once
and every pairwise comparison is arithmetic — the reference does not have to exist yet when
the candidates are run.

**`scripts/capability.py` — end-to-end capability.** 24 multi-step reasoning problems (each
needs several dependent steps, so a slightly drifted distribution derails somewhere) plus 8
code-generation tasks **verified by executing the generated function** against hidden asserts.

**Measured repeatability.** The sweep probed the same checkpoint (`unsloth/…-NVFP4`, MTP=3)
three times — twice on the V1 model runner and once on V2 — which is what every noise claim
above rests on:

| quantity | V1 run A | V1 run B | V2 run | V1↔V1 | V1↔V2 |
|---|---:|---:|---:|---:|---:|
| top-1 agree vs BF16 | 0.9565 | 0.9565 | 0.9565 | 0.000 | 0.000 |
| KL vs BF16 | 0.01978 | 0.01947 | 0.01923 | 1.6% rel. | 2.8% rel. |
| PPL (geomean) | 2.8798 | 2.8777 | 3.0758 | **0.073%** | **6.8%** |

Two distinct things, and conflating them is easy: the probe is *extremely* repeatable on a
fixed runner (PPL to 0.073%), and the **model runner itself** shifts likelihoods by 6.8% for
identical weights. Agreement is immune to both. So: rank freely within a runner, never carry
a PPL across runners, and never rank on a single probe position (individual positions do flip
between boots — two same-checkpoint runs agree with *each other* at 0.971 — even though the
aggregate rate against a common reference does not move at all).

Calibration notes, so the scores are read correctly:
- The reasoning set sits near this model's non-thinking ceiling — BF16 itself scores 14/24.
  That is deliberate: a saturated eval ranks nothing. It also means ±2 items is noise
  (σ≈2.4 on 24 binomial trials), so small gaps are not real. The sweep produced its own
  demonstration: `philbert440`'s 4-bit checkpoint scores **15/24, above unquantized BF16's
  14/24**. A 4-bit quant does not beat the model it was quantized from; that is what one
  standard deviation looks like, and it is why nothing here is ranked on this number alone.
- The code section scored 8/8 for **every** checkpoint including 4-bit. At this quantization
  range it is a floor check, not a discriminator.
- Consequently: lead with fidelity, corroborate with capability, and do not rank checkpoints
  on a 2-item capability difference alone.

### Benchmark caveats

Run-to-run variance ±5%; treat deltas <15% on the code probe as noise (its token count varies
with output length). The essay probe is the tightest discriminator (±1% within a boot). The
box drifts slightly under sustained load (power cap) — compare within a session. The
incumbent was re-measured in the same session as every challenger for exactly this reason,
and reproduced its previous numbers to within 1%.

### Repo layout

| Path | What |
|---|---|
| `scripts/serve.sh` | parametric container launcher (model, spec tokens, mem util, ctx, cpuset, YaRN) |
| `scripts/bench.py` | streamed decode benchmark; probes + concurrency ladder → JSON |
| `scripts/fidelity.py` | distributional fidelity vs a reference checkpoint (§8); `--compare` for offline ranking |
| `scripts/capability.py` | 24 multi-step reasoning + 8 execution-verified coding tasks |
| `scripts/quality.py` | original 12-question gate — kept as a fast boot check only (saturated) |
| `scripts/summarize.py` | regenerates every table in this README from `results/*.json` |
| `scripts/fetch_models.sh` | pulls candidate checkpoints into `hf-cache/` |
| `scripts/quant_sweep.sh` | one boot + full eval cycle per checkpoint |
| `scripts/run_experiment.sh` | one-command serve→bench→quality→fidelity→capability→stop |
| `scripts/phase*.sh`, `final_validation.sh` | the sweeps that produced the earlier tables |
| `dflash2-backport/` | vllm#52816 backported onto NGC 26.07, and why it still does not run |
| `results/*.json` | every measurement in this README, raw |
| `logs/` | server logs from the validated boots |
| `research/SOURCES.md` | third-party material consulted |

---

*Generated 2026-08-25 on gn100-6f38 (DGX Spark, driver 580.173.02, CUDA 13.3 userland /
580 kernel via forward-compat).*
