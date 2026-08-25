# Research sources

The raw snapshots that lived in this directory during the experiments are **not
committed** (third-party page dumps and vendored source; see `.gitignore`).
They were:

| Local snapshot | What it was |
|---|---|
| `ngc-vllm-running.html` | NVIDIA Docs — *Running vLLM* (DGX Spark / NGC container usage) |
| `vllm-recipe.html` | vLLM Recipes — *Qwen/Qwen3.8-27B* official recipe page |
| `spark-vllm-playbook.md` | NVIDIA DGX Spark playbook — *vLLM for Inference* |
| `sglang-recipe.md` | SGLang cookbook — *Qwen3.8-27B on SGLang for DGX Spark* (the cross-engine reference numbers cited in the README) |
| `qwen-bf16-card.md`, `qwen-bf16-config.json`, `qwen-fp8-config.json` | Hugging Face model card and `config.json` for `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` |
| `registry_main.py` | `vllm/model_executor/models/registry.py` from the vLLM main branch (Apache-2.0), used to confirm `Qwen3_5ForConditionalGeneration` / `Qwen3_5MTP` support in the NGC 26.07 build |

Everything in this repo outside this file — the recipe, scripts, results, and
logs — was produced on the machine described in the README.
