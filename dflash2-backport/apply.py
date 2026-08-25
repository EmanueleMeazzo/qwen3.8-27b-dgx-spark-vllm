#!/usr/bin/env python3
"""Backport vllm#52816 (DFlash2 drafter) into the NGC vLLM 26.07 container.

The NGC 26.07 image ships a vLLM snapshot that already has the DFlash *v1*
drafter but predates DFlash2 (merged upstream 2026-08-21). DFlash2 adds two
things v1 lacks -- per-layer two-tap grouped convolutions and a candidate
selector that walks a path through per-position top-k candidates -- so the
DFlash2 checkpoint cannot simply be loaded by the v1 code path.

The upstream change is pure Python (no CUDA kernels to rebuild), so it can be
applied in place. This script performs the edits to existing files; the new
modules are copied in by the Dockerfile.

Every edit is anchored on an exact snippet and asserts it matched, so a future
container whose sources have moved fails loudly instead of silently producing a
half-patched engine. Re-running is a no-op.
"""
import sys
from pathlib import Path

V = Path(sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm")


def edit(relpath: str, anchor: str, new: str, *, skip_if: str) -> None:
    p = V / relpath
    src = p.read_text()
    if skip_if in src:
        print(f"  skip (already applied)  {relpath}")
        return
    if src.count(anchor) != 1:
        raise SystemExit(
            f"FATAL: anchor for {relpath} matched {src.count(anchor)} times, expected 1.\n"
            f"The container's vLLM has moved; re-derive this patch.\n--- anchor ---\n{anchor}"
        )
    p.write_text(src.replace(anchor, new))
    print(f"  patched                 {relpath}")


# ---------------------------------------------------------------- registry
edit(
    "model_executor/models/registry.py",
    '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),',
    '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
    '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),',
    skip_if='"DFlash2DraftModel"',
)

# ------------------------------------------------- speculator dispatch
edit(
    "v1/worker/gpu/spec_decode/__init__.py",
    '    if speculative_config.method == "dflash":\n'
    "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (",
    '    if speculative_config.method == "dflash":\n'
    '        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:\n'
    "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
    "                DFlash2Speculator,\n"
    "            )\n"
    "\n"
    "            return DFlash2Speculator(vllm_config, device)\n"
    "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (",
    skip_if="DFlash2Speculator",
)

# ------------------------------------------------------------------ gumbel
# Standalone value-in/value-out Gumbel argmax. Upstream refactored the existing
# gumbel_block_argmax to share this; appending it instead leaves every existing
# sampling path byte-identical.
edit(
    "v1/worker/gpu/sample/gumbel.py",
    "@triton.jit\ndef gumbel_block_argmax(",
    '''@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # log1p keeps the winning tail at u -> 0, where fp32 resolves it.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(''',
    skip_if="def gumbel_noised_argmax",
)

# ------------------------------------------------------- base speculator hook
# DFlash2 writes only its K candidate columns per step, so the cache must start
# at -inf rather than 0. Routing the allocation through an overridable hook keeps
# every other speculator's buffer bit-identical (full(0.0, fp32) == zeros(fp32)).
edit(
    "v1/worker/gpu/spec_decode/speculator.py",
    "        self.draft_logits: torch.Tensor | None = None\n"
    '        if self.speculative_config.draft_sample_method == "probabilistic":\n'
    "            self.draft_logits = torch.zeros(\n"
    "                self.max_num_reqs,\n"
    "                self.num_speculative_steps,\n"
    "                self.vocab_size,\n"
    "                dtype=torch.float32,\n"
    "                device=device,\n"
    "            )",
    "        self.draft_logits: torch.Tensor | None = None\n"
    '        if self.speculative_config.draft_sample_method == "probabilistic":\n'
    "            dtype, fill = self.draft_logits_spec(vllm_config)\n"
    "            self.draft_logits = torch.full(\n"
    "                (\n"
    "                    self.max_num_reqs,\n"
    "                    self.num_speculative_steps,\n"
    "                    self.vocab_size,\n"
    "                ),\n"
    "                fill,\n"
    "                dtype=dtype,\n"
    "                device=device,\n"
    "            )\n"
    "\n"
    "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n"
    '        """Dtype and fill for the cached proposal distribution.\n'
    "\n"
    "        Speculators that write only a subset of columns each step override this.\n"
    '        """\n'
    "        return torch.float32, 0.0",
    skip_if="draft_logits_spec",
)

# --------------------------------------------------- dflash v1 subclass hooks
edit(
    "model_executor/models/qwen3_dflash.py",
    "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n    def __init__(",
    "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n"
    "    decoder_layer_cls = DFlashQwen3DecoderLayer\n\n"
    "    def __init__(",
    skip_if="decoder_layer_cls",
)
edit(
    "model_executor/models/qwen3_dflash.py",
    "                DFlashQwen3DecoderLayer(\n                    current_vllm_config,",
    "                self.decoder_layer_cls(\n                    current_vllm_config,",
    skip_if="self.decoder_layer_cls(",
)
edit(
    "model_executor/models/qwen3_dflash.py",
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n    def __init__(",
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    model_cls = DFlashQwen3Model\n\n"
    "    def __init__(",
    skip_if="model_cls = DFlashQwen3Model",
)
edit(
    "model_executor/models/qwen3_dflash.py",
    "        self.model = DFlashQwen3Model(",
    "        self.model = self.model_cls(",
    skip_if="self.model = self.model_cls(",
)

# ------------------------------------------------------- vocab-parallel top-k
edit(
    "model_executor/layers/logits_processor.py",
    "    def extra_repr(self) -> str:",
    '''    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits.

        The get_top_tokens reduction widened from one token to k, returning the
        values as well as the global ids. Scale and soft cap are applied to the k
        selected values rather than the whole vocabulary; both are monotonic, so
        the selection is unchanged and only k entries are touched.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        values, ids = torch.topk(logits, k, dim=-1)
        # Convert shard-local indices to global vocab indices.
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, k, dim=-1)
            ids = ids.gather(-1, selected)

        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            values = values * self.scale
        return ids, values

    def extra_repr(self) -> str:''',
    skip_if="def get_top_k_tokens",
)

print("dflash2 backport applied.")
