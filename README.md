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

### Faster: upstream vLLM + the DFlash2 drafter

The config above is the **NGC-supported** one. If you are not tied to NVIDIA's image, swapping
to upstream `vllm/vllm-openai:nightly` and the DFlash2 block drafter is dramatically faster at
equal quality on the right workload — **+82% on code, +35% on chat, +16% on prose** — see §4
for the exact command. DFlash2 does *not* work on the NGC image at all; that is a container
limitation, not a hardware one.

**It is not a blanket win, and the shape of it is now measured.** DFlash2's whole advantage is
that it emits more tokens per target forward pass, which happens only where text is predictable.
Pick it when your output is code-shaped, your context is short, and you serve a few streams at
a time; keep MTP otherwise:

| | DFlash2 (7) | MTP (3) |
|---|---|---|
| code, 1 stream | **54.1 tok/s** | 30.2 |
| prose, 1 stream | **22.9** | 20.4 |
| prose, 16 streams | 173.4 | **197.4** |
| prose @128K context | 17.4 | **19.4** |
| stability | 0 errors through 16 streams | 0 errors |

Both are stable under load; what changes is which one is faster. Details and the mechanism in
§4 (concurrency, per-position acceptance) and §5 (context depth).

**Two documented checkpoint alternatives** (§3): `RadixArk/Qwen3.8-27B-NVFP4` for ~+11% decode
at a measurably larger drift from the unquantized model, and `lued/Qwen3.8-27B-INT8-W8A16-MTP`
when you need BF16-identical output — it reproduces BF16's greedy choices exactly, at 1.53×
BF16's speed.

---

## 1. What this model is

`Qwen/Qwen3.8-27B` — arch `Qwen3_5ForConditionalGeneration` ("Qwen3.5" family):
- 64 layers: 48 GDN linear-attention + 16 full attention (`full_attention_interval: 4`)
- Vision tower included (serves images/video out of the box)
- Native 262,144-token context; extendable to 1M via YaRN (validated points: factor 2× @512K, 4× @1M).
  With MTP, set YaRN in `config.json`, **not** via `--hf-overrides` — the override never reaches
  the drafter and MTP acceptance drops to exactly 0 past 262K (§5)
- **In-checkpoint MTP draft layer** (`mtp_num_hidden_layers: 1`) — free speculative decoding, no extra download
- Thinking mode on by default; sampling defaults from `generation_config.json` (temp 1.0 / top_p 0.95 / top_k 20 thinking; 0.7 / 0.8 / 20 non-thinking)

## 2. Engine: container choice (measured)

| Container | Verdict |
|---|---|
| `nvcr.io/nvidia/vllm:25.12.post1-py3` (preinstalled here) | ❌ too old: no `Qwen3_5ForConditionalGeneration` |
| `vllm/vllm-openai:*` | arm64 images exist but NGC is the Spark-validated path |
| **`nvcr.io/nvidia/vllm:26.07-py3`** | ✅ vLLM 0.24.0-dev; has `Qwen3_5*`, `Qwen3_5MTP`, mtp/eagle/ngram/dflash spec methods |

**26.07 is the newest tag NVIDIA publishes** — verified against the NGC registry tag list
(`nvcr.io/v2/nvidia/vllm/tags/list`; nothing above 26.07 exists).

**But NGC is not the only option, and an earlier revision of this recipe was wrong to imply
it was.** Upstream `vllm/vllm-openai` publishes arm64 images, and they run on this box:

| | `nvcr.io/nvidia/vllm:26.07-py3` | `vllm/vllm-openai:nightly` |
|---|---|---|
| vLLM | 0.24.0+092c4842.dev (NVIDIA fork; commit not in public repo) | 0.26.1rc1.dev+ga9a17e709 |
| torch / CUDA | 2.13.0a0 / 13.x | 2.13.0+cu130 / 13.0 |
| compiled arch list | includes sm_121 | sm_80/90/100/110/**120** — no sm_121 |
| DFlash2 drafter | ✗ absent | ✅ present and complete |
| prefix caching | `False` — counters never even queried | `True` (~10× on a repeated long prefill, §5) |
| decode (unsloth NVFP4, MTP=3) | 28.0 / 19.2 / 22.2 | 28.3 / 19.9 / 21.7 |
| MTP acceptance | 0.600 | 0.622 |
| KV pool @ 0.90 | 2.33M tok | 2.25M tok |

The missing sm_121 turns out not to matter: CUDA guarantees binary compatibility forward
across minor revisions within a generation, so the sm_120 cubins execute on GB10, and
FlashInfer autotunes all 21 `fp4_gemm` profiles rather than falling back — the NVFP4
tensor-core path is fully live. Decode parity is within the ±5% run-to-run band.

So the two images are interchangeable for the MTP configuration, and upstream is strictly
ahead on features (see §4 — DFlash2 only works there). Prefer NGC if you need NVIDIA's
support path or a build that actually targets sm_121; prefer upstream if you want current
vLLM features.

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

**How the quality columns work** — see §9. `top-1 agree` is the fraction of probed positions
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
product question, not a benchmark one — the capability gate cannot resolve it (§9).

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


### DFlash2: the fastest drafter here — but only on upstream vLLM

`incoai/Qwen3.8-27B-DFlash2` is a block-diffusion drafter: it predicts a whole block of
tokens per pass and a learned selector traces one coherent path through the candidates.

**On upstream `vllm/vllm-openai:nightly` it works out of the box and is by far the fastest
configuration measured on this box:**

| config | code | essay | chat |
|---|---:|---:|---:|
| NGC 26.07 + MTP=3 | 28.0 | 19.2 | 22.2 |
| upstream + MTP=3 | 28.3 | 19.9 | 21.7 |
| **upstream + DFlash2 (7 draft tokens)** | **51.4** | **22.3** | **30.0** |
| gain over the MTP recommendation | **+83%** | **+16%** | **+35%** |

Those four rows come from three different boots on 2026-08-25. A later session re-measured the
DFlash2/MTP pair back to back on the same image, which is the comparison the drift caveat in §9
asks for:

| config (2026-08-27, one session) | code | essay | chat |
|---|---:|---:|---:|
| upstream + MTP=3 | 29.6 | 20.4 | 23.8 |
| **upstream + DFlash2 (7 draft tokens)** | **54.0** | **23.7** | **32.1** |
| gain | **+82%** | **+16%** | **+35%** |

Both sessions agree on the gains to within a point, even though the box ran 5–7% faster on the
second day across every probe — which is exactly why gains are quoted as ratios within a
session and never as absolute tok/s across sessions.

The mechanism is *not* higher accuracy: DFlash2's per-draft-token acceptance is **lower** than
MTP's on every probe. It wins because it drafts 7 tokens per verification step instead of 3,
so each expensive target forward pass emits more tokens.

**The content-dependence is now measured, and it is the whole story.** A previous revision
guessed that in-block acceptance is content-dependent but reported only a session-aggregate
0.385, which is an average over probes with wildly different behaviour. `bench.py` now
snapshots the spec-decode counters around each probe (`--spec-tokens k`), so acceptance is
attributed to the probe that produced it. Both drafters, same image, same boot:

| probe | DFlash2 accept | tok/target pass | MTP accept | tok/target pass | DFlash2 tok/s | MTP tok/s | gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0.717 | **6.02** | 0.818 | 3.46 | 54.0 | 29.6 | **+82%** |
| chat | 0.348 | 3.43 | 0.572 | 2.72 | 32.1 | 23.8 | +35% |
| essay | 0.231 | **2.62** | 0.470 | 2.41 | 23.7 | 20.4 | +16% |

Decode rate tracks tokens-per-target-pass almost exactly: DFlash2 emits 2.30× more tokens per
forward pass on code than on prose (6.02 vs 2.62) and runs 2.27× faster (54.0 vs 23.7). That
single quantity explains the entire spread, and the same-session MTP control reproduces the
+83%/+35%/+16% gains the earlier cross-session comparison reported.

The per-position vector says *why* — the fraction of draft blocks whose i-th token survives:

| probe | pos 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0.946 | 0.869 | 0.774 | 0.709 | 0.631 | 0.580 | **0.511** |
| chat | 0.794 | 0.608 | 0.412 | 0.258 | 0.186 | 0.103 | 0.072 |
| essay | 0.688 | 0.430 | 0.226 | 0.135 | 0.075 | 0.046 | **0.019** |

On code the seventh draft token is still right half the time; on prose it is right 1.9% of the
time, and positions 4–7 together contribute about a quarter of a token per pass. DFlash2 is not
"a faster drafter" — it is a drafter that sees far ahead in predictable text and barely one
token ahead in prose.

**Shrinking the block does not fix prose.** The obvious inference from that table is to stop
drafting where acceptance collapses, so the sweep re-ran DFlash2 with `num_speculative_tokens: 4`:

| probe | k=7 tok/s | k=4 tok/s | k=7 accept | k=4 accept | k=7 tok/pass | k=4 tok/pass |
|---|---:|---:|---:|---:|---:|---:|
| code | **54.0** | 39.2 | 0.717 | 0.855 | 6.02 | 4.42 |
| chat | **32.1** | 27.1 | 0.348 | 0.515 | 3.43 | 3.06 |
| essay | **23.7** | 22.4 | 0.231 | 0.381 | 2.62 | 2.52 |

Per-draft-token acceptance rises exactly as predicted (essay 0.231 → 0.381) and it buys
nothing: tokens per pass barely moves (2.62 → 2.52) and prose gets *slower*. The wasted draft
positions are nearly free, so trimming them removes upside without returning much compute.
Keep 7. (k=4 does have one use — under heavy concurrency; see below.)

Quality is unaffected on every measure taken: KL 0.0188 from unquantized BF16 versus 0.0198
for the MTP baseline, top-1 agreement 0.942 vs 0.957, and **22/32 on the capability gate —
the same score as unquantized BF16, unsloth NVFP4 and INT8**. No measurable cost for the speed.

Serve it with:

```bash
docker run -d --name vllm-qwen38 --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --cpuset-cpus 5-9,15-19 \
  -p 127.0.0.1:8000:8000 -v $HOME/dev/qwen3.8/hf-cache:/hf -e HF_HOME=/hf \
  vllm/vllm-openai:nightly \
    unsloth/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b --port 8000 \
    --max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

(The image's ENTRYPOINT is already `vllm serve`, so the model name comes first with no
`vllm serve` prefix. `scripts/serve.sh` handles both styles via `CMD_PREFIX`.)

Costs: the draft model adds ~3.3 GiB, and graph capture takes 108 s and 3.19 GiB, shrinking
the KV pool from 2.25M to 1.70M tokens (still 6.5× a full 262K request). Cold boot is ~6 min.

#### DFlash2 under concurrency: stable, but it gives the win back on prose

The previous revision recommended DFlash2 with the caveat that its numbers were single-stream
and its concurrency behaviour unknown. It is now measured, against a same-session MTP=3 control
on the same image (aggregate tok/s across all streams):

| streams | essay: DFlash2 | essay: MTP3 | ratio | code: DFlash2 | code: MTP3 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **22.9** | 20.4 | 1.12× | **54.1** | 30.2 | 1.79× |
| 2 | **44.3** | 38.9 | 1.14× | **90.9** | 54.9 | 1.66× |
| 4 | **74.5** | 67.5 | 1.10× | **165.8** | 103.8 | 1.60× |
| 8 | 119.2 | **125.1** | 0.95× | **263.6** | 184.8 | 1.43× |
| 16 | 173.4 | **197.4** | 0.88× | **350.5** | 284.5 | 1.23× |

**Zero errors at every rung for both drafters** — unlike `MTP=4`, which faults on the V1 runner
at ≥2 streams, DFlash2 is stable through 16 concurrent streams. The caveat was about missing
data, not instability.

But the advantage is load-dependent, and on prose it inverts: DFlash2 leads by 10–14% up to 4
streams, is 5% behind at 8 (inside the ±5% band, so call it even), and is **12% behind MTP at
16**. On code it leads everywhere, though the margin decays from 1.79× to 1.23×.

The drafter itself is not what degrades. Acceptance is flat on prose across the whole ladder
(0.225 → 0.246) and slips only 10% on code (0.741 → 0.669) while throughput per stream falls
by more than half. What decays is the *value* of drafting: batching
already amortizes the target forward pass, so speculation's benefit shrinks while its cost —
7 draft tokens of extra compute per step against MTP's 3 — does not. That cost is smallest
where the draft is most often right, which is why prose crosses over first.

This is the one place a shorter block pays. DFlash2 at `num_speculative_tokens: 4`, essay probe
(each rung reproduced 3×, spread ≤2%):

| streams | DFlash2 k=7 | DFlash2 k=4 | MTP3 |
|---:|---:|---:|---:|
| 8 | **119.2** | 104.8 | **125.1** |
| 16 | 173.4 | **196.6** | 197.4 |

At 16 streams k=4 recovers the whole deficit and matches MTP (196.6 vs 197.4); at 8 streams it
is the worst of the three. Both numbers reproduce, so the non-monotonicity is real rather than
noise, and this recipe does not have an explanation for it — the honest summary is that k=4 is
worth trying only if you are running prose at high concurrency and have measured your own mix.

**On NGC 26.07 it cannot work**, and `dflash2-backport/` documents that in detail. The image
ships DFlash *v1*; DFlash2's convolutions and selector landed upstream on 2026-08-21, after
the image was cut. The backport applies the upstream change (pure Python), and gets as far as
loading the draft, compiling the selector and completing graph capture — then faults in
`propose()`, because the container's v1 speculator pads `sample_idx_mapping` with `0` where
DFlash2's kernel uses `valid = req_state >= 0` as its padding test, and stores `sample_pos`
un-incremented. Those were left unpatched deliberately: guessing the buffer convention would
stop the crash while silently mis-sampling drafts, and lossless speculation that is not
lossless is worse than none. **Use upstream instead — that directory is only relevant if you
are pinned to NGC.**

Rejected alternatives:
- **TRITON_ATTN backend**: identical perf at bs=1 despite enabling FULL_AND_PIECEWISE graphs.
- **ngram**: SGLang box measured it ~30% under MTP on this same model; skipped.

Sampling robustness: greedy vs temp 0.7 differ by <3% throughput; thinking mode
(temp 1.0) costs ~25% decode speed. Use thinking when quality demands it, not by default.

## 5. Long context (measured)

Every other number in this recipe comes from a prompt of about 100 tokens, while the headline
claim is a 262,144-token context. Those are different questions, and nothing here answered the
second one until now. `scripts/bench.py --depths ... --needle` sizes a real, non-repetitive
excerpt (Moby-Dick, fetched on first use) to a target token count, buries a unique fact at 60%
depth, and measures prefill, decode, drafter acceptance and whether the model still finds the
fact. Tiled or generated filler would be far more predictable than natural text and would
inflate acceptance, so the probe uses real prose.

**Prefill is the cost that matters at depth.** Cold TTFT, MTP=3 boot:

| context | prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|---:|
| 1K | 1,097 | 0.55 s | 2,002 |
| 4K | 4,186 | 1.81 s | 2,308 |
| 16K | 16,442 | 12.0 s | 1,365 |
| 64K | 65,484 | 59.0 s | 1,110 |
| 128K | 130,752 | **150.2 s** | 871 |

Prefill throughput degrades by ~2.6× from 4K to 128K, and because the token count grows faster
than the rate falls, TTFT grows superlinearly: **a 128K-token prompt costs two and a half
minutes before the first token appears.** Budget for that in any agentic loop that re-sends a
large context. The DFlash2 boot traces the same curve (0.64 s → 145.9 s), agreeing to
within 3–7% from 16K up; the two shallow points differ by ~20%, which is sub-second
absolute and dominated by per-request overhead rather than prefill.

**Retrieval holds.** The needle was recovered exactly at every depth on both drafters, 1K
through 128K — the context is usable, not merely allocated.

**The two drafters move in opposite directions with depth**, which is the most consequential
finding in this section:

| context | DFlash2 decode | MTP3 decode | DFlash2 accept | MTP3 accept |
|---:|---:|---:|---:|---:|
| 1K | **28.4** | 19.8 | 0.316 | 0.444 |
| 4K | **24.1** | 20.9 | 0.248 | 0.489 |
| 16K | 22.1 | 21.8 | 0.223 | 0.533 |
| 64K | 19.8 | **20.2** | 0.199 | 0.536 |
| 128K | 17.4 | **19.4** | 0.186 | 0.573 |

DFlash2 loses 39% of its decode rate between 1K and 128K; MTP loses 2%. The acceptance columns
show the cause and rule out a shared attention-cost explanation: on identical prompts DFlash2's
acceptance **falls** (0.316 → 0.186) while MTP's **rises** (0.444 → 0.573). More context makes
the single-token MTP head a better predictor; it makes the block drafter's later positions
worse. They cross over between 16K and 64K.

Note the confound: the depth probe generates prose, so DFlash2 starts from its weakest content
type. The depth effect is still separable — both drafters are measured on the same prompts, and
only one of them degrades — but the crossover point would sit deeper for code-shaped output.

**Prefix caching is on upstream and off on NGC — and it needs the same prompt twice.** §6's
table used to record prefix caching as unavailable for this hybrid architecture. That is true
of the NGC build and false of upstream: across every boot in `logs/`, NGC 26.07 reports
`enable_prefix_caching=False` and upstream nightly (0.26.1rc1.dev1177) reports `True`.

A dedicated NGC boot confirms the flag behaviourally rather than by configuration alone. The
same 32,894-token prompt, served three times:

| serving | NGC 26.07 TTFT | prefix-cache queries | upstream TTFT | queries |
|---|---:|---:|---:|---:|
| first (cold) | 27.12 s | **0** | 24.14 s | 32,903 |
| second | 26.90 s | **0** | 24.28 s | 32,903 |
| third | 26.99 s | **0** | **2.39 s** | 32,903 |

On NGC the cache is not merely unhelpful, it is never consulted — the counters stay at zero and
all three prefills cost full price. (Incidentally, NGC's cold prefill of the same prompt is
~11% slower than upstream's, 26.9–27.1 s against 24.1–24.3 s, tight across repeats on both.)

What it delivers is not what the flag suggests. Serving an identical 32,891-token prompt three
times in a row:

| serving | TTFT | prefix blocks hit |
|---|---:|---:|
| first (cold) | 24.14 s | 0.0% |
| second | 24.28 s | 0.0% |
| third | **2.39 s** | 90.2% |

The *first* repeat gets nothing; the second gets a 10.1× faster prefill. Inserting a 10-second
pause before the repeat does not help, so this is not a race against block release — the prefix
has to be served twice before it becomes reusable. Worth knowing before you design a caching
strategy around it, and worth measuring on your own workload rather than assuming the flag is
free.

This also invalidates any depth measurement that reuses a prompt: an earlier run of this sweep
reported 37,000 tok/s prefill at 128K, which is simply a cache hit. Every timed prompt in
`bench.py` now carries a unique nonce at position 0.

### Beyond 262K: YaRN to 512K, and a trap for MTP

Everything above stops at 128K on upstream. `scripts/phase8.sh` measures the NGC recipe itself
(26.07, unsloth NVFP4, MTP=3) out to ~496K with YaRN factor 2, three boots in one session:
native RoPE as the control, YaRN via `--hf-overrides` exactly as the model card and the
previous version of this README said, and YaRN written into `config.json`. Moby-Dick alone is
~340K tokens, so the depth corpus now continues into War and Peace; every prompt up to 128K is
byte-identical to the one §5 measured above.

**At short context YaRN costs nothing.** Code / essay / chat decode: 28.3 / 19.3 / 22.4 tok/s
native, 28.9 / 19.5 / 22.8 with `--hf-overrides`, 29.6 / 20.6 / 22.8 with `config.json` YaRN —
all within boot-to-boot noise, although vLLM applies the rescaled RoPE to every request.

**Past 262K, `--hf-overrides` switches MTP off — and then it costs you.** Prose decode at depth
(one 300-token generation per point, so ±10% is noise below 262K):

| context | native | YaRN, `--hf-overrides` | accept | YaRN, `config.json` | accept | TTFT (cold) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 19.5 | 19.5 | 0.460 | 22.3 | 0.555 | 0.7 s |
| 16K | 22.3 | 19.3 | 0.473 | 20.3 | 0.484 | 13 s |
| 128K | 18.0 | 17.5 | 0.511 | 19.9 | 0.590 | 151 s |
| 256K | — | **7.3** | **0.000** | **16.5** | 0.559 | 425 s |
| ~496K | — | **5.8** | **0.000** | **12.2** | 0.496 | **1,255 s** |

With `--hf-overrides` the drafter accepted 0 of 897 drafted tokens at both deep points — not a
degradation, a cliff exactly at the native limit — and decode fell *below* the 11.2 tok/s
no-spec reference, because the engine still pays for three dead drafts per step. The cause is in
vLLM (`vllm/config/speculative.py`, NGC 26.07 ≈ v0.24): the MTP draft model's `ModelConfig` is
rebuilt from the checkpoint with `hf_overrides=SpeculativeConfig.hf_config_override`, so the
user's override never reaches it. The drafter keeps native RoPE and a 262,144-token
`max_model_len`, and past that its predictions are noise. The target model is fine throughout:
the needle was found at every depth on every boot.

Writing YaRN into the checkpoint's own `config.json` gives target and drafter the same RoPE
(verified in-container through the same `hf_config_override` path the drafter uses). Acceptance
then holds at 0.50–0.56 out to ~496K, and what remains is the real cost of depth: **12.2 tok/s at
~496K, ~40% below the 1K–128K band on prose.** The recipe, with a symlinked copy so the weights
are not duplicated:

```bash
cd hf-cache
snap=hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/<rev>
dst=local/Qwen3.8-27B-NVFP4-yarn2; mkdir -p $dst
for f in $snap/*; do [ "$(basename $f)" = config.json ] || ln -sfn ../../$f $dst/; done
python3 - $snap/config.json $dst/config.json <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c)
t["rope_parameters"].update(rope_type="yarn", factor=2.0, original_max_position_embeddings=262144)
t["max_position_embeddings"] = 524288
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
# then serve /hf/local/Qwen3.8-27B-NVFP4-yarn2 with --max-model-len 524288, no --hf-overrides
```

Prefill, not decode, is what makes 512K expensive: 617 tok/s at 256K and ~400 tok/s at ~496K,
so a cold ~500K prompt costs **~21 minutes** before the first token, and NGC has no prefix
caching to amortise it. Not measured: the code probe at depth (the ~28 tok/s headline is code;
the depth probe is prose), and factor 4 / 1M.

---

## 6. Serving parameters (measured)

| Flag | Value | Why |
|---|---|---|
| `--gpu-memory-utilization` | **0.90** | 0.85→KV 2.11M tok; 0.90→2.33M tok, no instability. Do NOT exceed ~0.92: OOM on unified memory can hard-wedge the box |
| `--max-num-seqs` | 16 | matches concurrency ladder; capping to 4 queues badly at 8+ streams (TTFT 20–42 s) |
| `--max-num-batched-tokens` | 8192 | good TTFT/throughput balance at these prompt sizes |
| `--kv-cache-dtype fp8` | yes | ~2× KV density; standard on this model |
| `--max-model-len` | 262144 | native, and retrieval is intact at 128K (§5) — but a 128K prompt costs ~150 s of prefill. For 512K: YaRN factor 2 **written into `config.json`** of a local checkpoint copy (§5), not `--hf-overrides`, which leaves the MTP drafter on native RoPE (0% acceptance past 262K). A ~500K cold prompt costs ~21 min of prefill |
| `--cpuset-cpus 5-9,15-19` | X925 cores only | big.LITTLE box; scheduler/tokenizer never land on A725 efficiency cores (SGLang box: +2–7% decode) |
| attention backend | default (FlashInfer) | TRITON_ATTN measured identical |
| prefix caching | image-dependent | NGC 26.07 reports `enable_prefix_caching=False`; upstream nightly reports `True` and delivers ~10× on a repeated long prefill — but only from the *second* repeat (§5) |

## 7. Operational notes

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
  confidently wrong. `scripts/fidelity.py` routes around it (§9). Suspected cause is the
  hybrid GDN layers, whose recurrent state is only correct at the final position.

## 8. Reproduce / re-tune everything in this repo

```bash
git clone https://github.com/EmanueleMeazzo/qwen3.8-27b-dgx-spark-vllm.git
cd qwen3.8-27b-dgx-spark-vllm

# full experiment: serve -> wait -> bench -> quality -> fidelity -> capability -> stop
scripts/run_experiment.sh myrun "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90"

# the quantization sweep in this README
scripts/fetch_models.sh RadixArk/Qwen3.8-27B-NVFP4 RedHatAI/Qwen3.8-27B-INT4
scripts/quant_sweep.sh "Q-radixark-nvfp4=RadixArk/Qwen3.8-27B-NVFP4"

# drafter profiling: per-probe acceptance, concurrency ladders, context depth (§4, §5)
scripts/phase6.sh

# regenerate every table above from results/*.json
scripts/summarize.py --ref Q-bf16
```

Weights and the vLLM compile cache land in `hf-cache/` (git-ignored, ~180 GB after the full
sweep). Every knob in `scripts/serve.sh` is an env var — see the header of that file.

## 9. How quality is measured

The original `quality.py` — twelve one-hop questions like "what is 17×23" — scored **12/12 for
every checkpoint that booted at all**. It is a breakage detector, not a ranking, and it is
kept only as a fast boot gate. Ranking quants needs two things it cannot do:

**`scripts/fidelity.py` — distributional fidelity.** Probes ~69 fixed positions across four
passages (English prose, Python, worked mathematics, Chinese) and reports, per position, the
checkpoint's next-token distribution. Against a reference checkpoint it yields `top-1 agree`
(how often greedy decoding would pick the same token as unquantized BF16) and `KL(ref‖x)`.
This is the sensitive instrument: 69 positions, deterministic, no sampling.

It deliberately does **not** use `echo`/prompt-logprobs, which is broken here (§7). Every
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

Across *days* the drift is larger than the within-session band: re-running the identical
DFlash2 configuration two days later gave +5.1% (code), +6.5% (essay) and +6.9% (chat). The
consistent sign across all three probes marks it as a session-level shift in the box rather
than probe noise, and it is the reason every cross-configuration claim in §4 and §5 rests on a
control booted in the same session rather than on a number carried over from an earlier table.

### Repo layout

| Path | What |
|---|---|
| `scripts/serve.sh` | parametric container launcher (model, spec tokens, mem util, ctx, cpuset, YaRN) |
| `scripts/bench.py` | streamed decode benchmark; probes, per-probe/per-position drafter acceptance, concurrency ladder, context-depth sweep + needle → JSON |
| `scripts/fidelity.py` | distributional fidelity vs a reference checkpoint (§9); `--compare` for offline ranking |
| `scripts/capability.py` | 24 multi-step reasoning + 8 execution-verified coding tasks |
| `scripts/quality.py` | original 12-question gate — kept as a fast boot check only (saturated) |
| `scripts/summarize.py` | regenerates every table in this README from `results/*.json` |
| `scripts/fetch_models.sh` | pulls candidate checkpoints into `hf-cache/` |
| `scripts/quant_sweep.sh` | one boot + full eval cycle per checkpoint |
| `scripts/run_experiment.sh` | one-command serve→bench→quality→fidelity→capability→stop |
| `scripts/phase6.sh` | drafter profiling: per-probe acceptance, concurrency ladders and the depth sweep (§4, §5) |
| `scripts/phase8.sh` | YaRN 512K on NGC: native vs `--hf-overrides` vs `config.json` YaRN, depths to ~496K (§5) |
| `scripts/phase*.sh`, `final_validation.sh` | the sweeps that produced the earlier tables |
| `dflash2-backport/` | vllm#52816 backported onto NGC 26.07, and why it still does not run |
| `results/*.json` | every measurement in this README, raw |
| `logs/` | server logs from the validated boots |
| `research/SOURCES.md` | third-party material consulted |

---

*Generated 2026-08-25 on gn100-6f38 (DGX Spark, driver 580.173.02, CUDA 13.3 userland /
580 kernel via forward-compat). §4 concurrency/acceptance, §5 long context, and the
prefix-caching correction added 2026-08-27 from the same box; the 512K/YaRN measurements in §5
added 2026-11-01.*
