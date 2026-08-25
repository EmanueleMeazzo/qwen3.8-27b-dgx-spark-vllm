# DFlash2 drafter on NGC vLLM 26.07 — backport

## Why a backport at all

`nvcr.io/nvidia/vllm:26.07-py3` is the **newest** vLLM container NVIDIA publishes
(verified against the NGC registry tag list — there is no 26.08+). So "update the
container to get DFlash2" is not an available move.

It is also not quite the right diagnosis. The 26.07 image **already ships the DFlash
v1 drafter** — `vllm/v1/worker/gpu/spec_decode/dflash/`, plus a `DFlashDraftModel`
registry entry and a `"dflash"` speculative method. What it lacks is DFlash**2**,
which adds two things v1 has no code for:

* per-layer **two-tap grouped convolutions** around attention and MLP
  (`attention_conv`, `mlp_conv` — 20 tensors the v1 loader would reject), and
* a **candidate selector** (`predecessor_codebook` / `successor_codebook` /
  `hidden_projection`) that walks one path through per-position top-k candidates.

`incoai/Qwen3.8-27B-DFlash2` therefore declares `architectures: ["DFlash2DraftModel"]`,
which the container's registry does not know.

Upstream added it in **[vllm#52816](https://github.com/vllm-project/vllm/pull/52816)**,
merged **2026-08-21** — after this image was cut. The change is **pure Python: no CUDA
kernels, no rebuild**, so it can be applied to the installed package in place.

## What this directory does

`docker build -t vllm-dflash2:26.07 dflash2-backport/` — seconds, since every compiled
artifact is inherited from the base image.

Two new modules are copied in whole (`qwen3_dflash2.py`, `spec_decode/dflash2/`), and
`apply.py` makes six anchored in-place edits: the registry entry, the speculator
dispatch, a standalone `gumbel_noised_argmax`, an overridable `draft_logits_spec` hook,
`decoder_layer_cls`/`model_cls` subclass hooks on the v1 dflash model, and a
vocab-parallel `get_top_k_tokens`.

Every edit asserts its anchor matched exactly once, so a container whose sources have
moved fails loudly instead of silently producing a half-patched engine. Re-running is a
no-op. The edits are behaviour-preserving for every existing speculator: the only change
to a shared path swaps `torch.zeros(...)` for an equivalent `torch.full(..., 0.0)`.

Two adaptations were needed because the container's tree is an NVIDIA fork that predates
the PR's base:

* the container's `DFlashQwen3DecoderLayer` has no `layer_idx` parameter (causality is
  resolved globally by `get_dflash_causal`, not per layer), so the DFlash2 override drops it;
* `get_top_k_tokens` is rebuilt on the container's `get_top_tokens` (which uses
  `lm_head.quant_method.apply` and `torch.topk`) rather than upstream's FlashInfer radix top-k.

DFlash2's selector only exists on the V2 model runner, so serve with
`-e VLLM_USE_V2_MODEL_RUNNER=1`.

## Status on this box: loads and initializes, but drafting faults

```
Resolved architecture: DFlash2DraftModel          <- registry patch works
Using V2 Model Runner
Model loading took 27.48 GiB                      <- target + DFlash2 draft, no weight errors
.../torch_compile_cache/.../dflash2_candidate_selector   <- selector compiles
GPU KV cache size: 1,528,478 tokens
Graph capturing finished in 9 secs
EngineCore failed to start.
  ... dflash/speculator.py:346 in propose
  torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Everything structural works. The fault is in `propose()`, and the reason is the part of
the gap a pure-Python drop-in cannot close. `DFlash2Speculator` subclasses the
container's `DFlashSpeculator`, and that file is **~376 lines behind** the version
DFlash2's kernels were written against. Diffing the two shows the two conventions
DFlash2's Triton kernels encode, both of which the container's v1 speculator predates:

| | container (v1, 26.07) | upstream (DFlash2's base) |
|---|---|---|
| `sample_idx_mapping` init / reset / pad | `zeros(...)`, `.zero_()`, pads with **0** | `full(..., -1)`, `.fill_(-1)`, pads with **-1** |
| `sample_pos` stored | `query_pos` | `query_pos + 1`, and the consumer subtracts 2 |

`_selector_walk_kernel` opens with

```python
req_state = tl.load(req_state_ptr + row * num_steps)
valid = req_state >= 0
```

which **is** the padding test — it relies on the `-1` sentinel. Against the container's
buffer every padded row reads back `0`, so `valid` is true for rows that were never
populated, and the kernel goes on to index `temperature_ptr`/`seeds_ptr` with them. That
is the illegal access. The `sample_pos` skew is the quieter half: DFlash2 computes its
Gumbel noise position as `sample_pos - 1` against a buffer that upstream stores
pre-incremented, so even once it stops faulting the draft would be keyed off-by-one and
silently mis-sampled.

Both could be adapted — but "make the crash stop" here is exactly the dangerous fix: the
kernel would then run against buffers whose meaning it has guessed, and a wrong draft
distribution is worse than no DFlash2, because speculative decoding is supposed to be
lossless and nothing downstream would flag it. Doing it properly means backporting
`dflash/speculator.py` as well, which in turn wants base `DraftModelSpeculator` methods
this container does not have (`attn_vllm_config`, `_validate_local_argmax_reduction`,
`set_eplb_state`, …) — a cascade, not a patch.

**Conclusion: DFlash2 needs a vLLM newer than any container NVIDIA currently ships.**
Keep the in-checkpoint MTP head as the drafter until NGC publishes an image built after
2026-08-21. This directory is what to re-run when that image lands: point `BASE` at it,
and `apply.py` will either apply cleanly or name the exact anchor that moved.
