# Research sources

The raw snapshots that lived in this directory during the experiments are **not
committed** (third-party page dumps and vendored source; see `.gitignore`).

## First round (initial recipe)

| Local snapshot | What it was |
|---|---|
| `ngc-vllm-running.html` | NVIDIA Docs — *Running vLLM* (DGX Spark / NGC container usage) |
| `vllm-recipe.html` | vLLM Recipes — *Qwen/Qwen3.8-27B* official recipe page |
| `spark-vllm-playbook.md` | NVIDIA DGX Spark playbook — *vLLM for Inference* |
| `sglang-recipe.md` | SGLang cookbook — *Qwen3.8-27B on SGLang for DGX Spark* (the cross-engine reference numbers cited in the README) |
| `qwen-bf16-card.md`, `qwen-bf16-config.json`, `qwen-fp8-config.json` | Hugging Face model card and `config.json` for `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` |
| `registry_main.py` | `vllm/model_executor/models/registry.py` from the vLLM main branch (Apache-2.0), used to confirm `Qwen3_5ForConditionalGeneration` / `Qwen3_5MTP` support in the NGC 26.07 build |

## Second round (quantization sweep + DFlash2 backport)

Queried live rather than snapshotted; each is reproducible from the command shown.

| Source | Used for |
|---|---|
| `nvcr.io/v2/nvidia/vllm/tags/list` (NGC registry API) | Confirming **26.07-py3 is the newest published tag** — there is no newer container to upgrade to |
| Hugging Face model API (`/api/models?search=Qwen3.8-27B`, `?blobs=true`) | Enumerating community quantizations and their exact `.safetensors` byte sizes |
| `config.json` + `model.safetensors.index.json` of each candidate | Verifying every candidate retains the in-checkpoint MTP head (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors) **before** downloading it |
| GitHub API — `vllm-project/vllm` PR **#52816** ("[Spec Decode] DFlash2: local convolution + candidate selector"), merged 2026-08-21 | The DFlash2 drafter backport in `dflash2-backport/` |
| `raw.githubusercontent.com` at the PR's merge commit and its first parent | Deriving the pre/post images used for the three-way merge onto the container's (NVIDIA-fork) vLLM tree |
| `incoai/Qwen3.8-27B-DFlash2` model card and `config.json` | DFlash2 drafter block size, selector rank/top-k, and the upstream-recommended `num_speculative_tokens: 7` |

Everything in this repo outside this file — the recipe, scripts, results, and
logs — was produced on the machine described in the README.
