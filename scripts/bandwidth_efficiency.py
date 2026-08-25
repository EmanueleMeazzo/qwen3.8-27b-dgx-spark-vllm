#!/usr/bin/env python3
"""Bandwidth efficiency: actual decode rate vs what weight bytes alone predict.

Decode on this box is LPDDR5X-bandwidth-bound, so a checkpoint's speed *should* scale
as 1/bytes. It does -- but only within a kernel family. This prints the ratio so the
exception is visible rather than inferred.

Baseline is the recommended checkpoint; predicted = baseline_rate * baseline_bytes / bytes.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = [
    ("unsloth NVFP4",  "Q-unsloth-nvfp4",  21.8, "native FP4"),
    ("RadixArk NVFP4", "Q-radixark-nvfp4", 20.4, "native FP4"),
    ("BF16",           "Q-bf16",           51.7, "native bf16"),
    ("RedHatAI INT4",  "Q-redhat-int4",    18.1, "W4A16 dequant"),
    ("philbert AWQ",   "Q-philbert-awq",   18.2, "W4A16 dequant"),
    ("lued INT8",      "Q-lued-int8",      29.4, "W8A16 dequant"),
]
BASE = "Q-unsloth-nvfp4"


def essay(tag):
    p = os.path.join(ROOT, "results", f"{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["probes"]["essay"]["median"]["net_decode_tok_s"]


def main():
    base_rate = essay(BASE)
    base_bytes = dict((t, g) for _, t, g, _ in ROWS)[BASE]
    if base_rate is None:
        sys.exit(f"missing baseline {BASE}")
    print("| Checkpoint | Weights | essay tok/s | predicted from bytes | achieved | GEMM path |")
    print("|---|---:|---:|---:|---:|---|")
    for name, tag, gib, path in ROWS:
        r = essay(tag)
        if r is None:
            print(f"| {name} | {gib} GiB | — | — | — | {path} |")
            continue
        pred = base_rate * base_bytes / gib
        print(f"| {name} | {gib} GiB | {r:.2f} | {pred:.2f} | **{r / pred:.2f}×** | {path} |")


if __name__ == "__main__":
    main()
