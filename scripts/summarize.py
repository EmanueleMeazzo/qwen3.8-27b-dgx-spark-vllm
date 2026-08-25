#!/usr/bin/env python3
"""Assemble the README's comparison tables from results/*.json.

Hand-copying two dozen numbers out of JSON into markdown is how recipes end up with
tables that no longer match their own data. This regenerates them.

Usage:
  python3 summarize.py                       # speed + quality tables
  python3 summarize.py --ref Q-bf16           # score fidelity against a reference tag
"""
import argparse, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# tag -> (display name, checkpoint, weight GiB). Weight sizes are the summed
# .safetensors bytes reported by the HF API, i.e. what the decoder must stream per token.
CHECKPOINTS = [
    ("Q-unsloth-nvfp4",   "unsloth/Qwen3.8-27B-NVFP4",                       "NVFP4 (mixed FP8+FP4)", 21.8),
    ("Q-radixark-nvfp4",  "RadixArk/Qwen3.8-27B-NVFP4",                      "NVFP4 (MTP unquantized)", 20.4),
    ("Q-redhat-int4",     "RedHatAI/Qwen3.8-27B-INT4",                       "INT4 W4A16", 18.1),
    ("Q-philbert-awq",    "philbert440/Qwen3.8-27B-W4A16-AWQ",               "INT4 W4A16 AWQ", 18.2),
    ("Q-lued-int8",       "lued/Qwen3.8-27B-INT8-W8A16-MTP",                 "INT8 W8A16", 29.4),
    ("Q-bf16",            "Qwen/Qwen3.8-27B",                                "BF16 (unquantized)", 51.7),
]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def speed_table():
    print("| Checkpoint | Scheme | Weights | code | essay | chat | TTFT | MTP accept |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, ckpt, scheme, gib in CHECKPOINTS:
        d = load(os.path.join(R, f"{tag}.json"))
        if not d:
            print(f"| `{ckpt}` | {scheme} | {gib} GiB | — | — | — | — | *(not measured)* |")
            continue
        p = d["probes"]
        m = d.get("mtp") or {}
        acc = m.get("accepted_over_draft")
        ttft = p["chat"]["median"]["ttft_s"]
        print(f"| `{ckpt}` | {scheme} | {gib} GiB | "
              f"{p['code']['median']['net_decode_tok_s']:.1f} | "
              f"{p['essay']['median']['net_decode_tok_s']:.1f} | "
              f"{p['chat']['median']['net_decode_tok_s']:.1f} | "
              f"{ttft:.2f} s | {acc if acc is None else f'{acc:.3f}'} |")


def quality_table(ref_tag):
    ref = load(os.path.join(R, f"fid-{ref_tag}.json"))
    print()
    hdr = "| Checkpoint | PPL | top-1 agree vs ref | KL(ref‖x) | reasoning | code |"
    print(hdr)
    print("|---|---:|---:|---:|---:|---:|")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from fidelity import kl_ref_vs
    for tag, ckpt, _scheme, _gib in CHECKPOINTS:
        fid = load(os.path.join(R, f"fid-{tag}.json"))
        cap = load(os.path.join(R, f"cap-{tag}.json"))
        if not fid:
            print(f"| `{ckpt}` | — | — | — | — | — |")
            continue
        agree = n = 0
        kls = []
        if ref and tag != ref_tag:
            for name, rec in fid["passages"].items():
                rr = ref["passages"].get(name)
                if not rr:
                    continue
                for a, b in zip(rec["probes"], rr["probes"]):
                    if a["pos"] != b["pos"] or a["target"] != b["target"]:
                        continue
                    n += 1
                    agree += a["top1"] == b["top1"]
                    kls.append(kl_ref_vs(b["dist"], a["dist"]))
        agree_s = f"{agree / n:.3f}" if n else ("*reference*" if tag == ref_tag else "—")
        kl_s = f"{sum(kls) / len(kls):.4f}" if kls else ("*0*" if tag == ref_tag else "—")
        rea = f"{cap['reasoning']['score']}/{cap['reasoning']['total']}" if cap and "reasoning" in cap else "—"
        cod = f"{cap['code']['score']}/{cap['code']['total']}" if cap and "code" in cap else "—"
        print(f"| `{ckpt}` | {fid.get('ppl_geomean', float('nan')):.3f} | {agree_s} | {kl_s} | {rea} | {cod} |")


def ladders():
    print()
    for fn in sorted(os.listdir(R)):
        if not fn.endswith("-ladder.json"):
            continue
        d = load(os.path.join(R, fn))
        rows = (d or {}).get("ladder") or []
        if not rows:
            continue
        print(f"**{fn[:-5]}**: " + " · ".join(
            f"{r['streams']}→{r['aggregate_tok_s']} tok/s"
            + (f" ({r['errors']} err)" if r.get("errors") else "")
            for r in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Q-bf16", help="tag used as the fidelity reference")
    a = ap.parse_args()
    speed_table()
    quality_table(a.ref)
    ladders()
