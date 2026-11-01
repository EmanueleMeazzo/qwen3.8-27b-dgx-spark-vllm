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


# ---- phase 6: the three quantities §4/§5 left unmeasured ---------------------
# Tags are P6-<drafter>-{probes,ladder,depth}; drafter order is fixed so the
# control (MTP=3) always prints under the challenger.
P6 = [("P6-dflash2", "DFlash2 (7 draft tok)"),
      ("P6-dflash2k4", "DFlash2 (4 draft tok)"),
      ("P6-mtp3", "MTP (3 draft tok)")]

# Phase 7 — vLLM 0.28.0 against a same-session 0.26.1 control. Kept as a separate group
# because the version, not just the drafter, differs between rows: never read a 0.28.0
# row against a P6 row (different session, and §9's cross-day drift is 5-7%).
P7 = [("P7-new-mtp3",    "0.28.0 · MTP (3)"),
      ("P7-old-mtp3",    "0.26.1 · MTP (3)"),
      ("P7-new-dflash2", "0.28.0 · DFlash2 (7)"),
      ("P7-old-dflash2", "0.26.1 · DFlash2 (7)"),
      ("P7-new-dspark",  "0.28.0 · DSpark (7)")]

# Phase 8: YaRN x2 (512K) vs native RoPE, NGC 26.07 + MTP=3, same session.
P8 = [("P8-ctl",  "native RoPE · 262K"),
      ("P8-yarn", "YaRN x2 · 512K (--hf-overrides)"),
      ("P8-yarncfg", "YaRN x2 · 512K (config.json)")]


def _acc(spec):
    return "—" if not spec else f"{spec['accepted_over_draft']:.3f}"


def _tpp(spec):
    if not spec or "tokens_per_target_pass" not in spec:
        return "—"
    return f"{spec['tokens_per_target_pass']:.2f}"


def per_probe_acceptance(group=P6):
    """Is draft acceptance content-dependent? One row per drafter x probe."""
    print("\n| Drafter | Probe | decode tok/s | accept/draft | tokens per target pass |")
    print("|---|---|---:|---:|---:|")
    for tag, label in group:
        d = load(os.path.join(R, f"{tag}-probes.json"))
        if not d:
            continue
        for probe in ("code", "essay", "chat"):
            pr = d["probes"].get(probe)
            if not pr:
                continue
            print(f"| {label} | {probe} | {pr['median']['net_decode_tok_s']:.1f} | "
                  f"{_acc(pr.get('spec'))} | {_tpp(pr.get('spec'))} |")


def concurrency_by_probe(group=P6):
    """Aggregate throughput and acceptance per rung, prose vs code."""
    for tag, label in group:
        d = load(os.path.join(R, f"{tag}-ladder.json"))
        lads = (d or {}).get("ladders") or {}
        if not lads:
            continue
        probes = [p for p in ("essay", "code") if p in lads]
        print(f"\n**{label}**\n")
        print("| streams | " + " | ".join(
            f"{p} agg tok/s | {p} accept" for p in probes) + " | errors |")
        print("|---:|" + "---:|" * (2 * len(probes)) + "---:|")
        for i, rung in enumerate(lads[probes[0]]):
            cells, errs = [], 0
            for p in probes:
                r = lads[p][i]
                errs += r.get("errors", 0)
                cells += [f"{r['aggregate_tok_s']:.0f}", _acc(r.get("spec"))]
            print(f"| {rung['streams']} | " + " | ".join(cells) + f" | {errs} |")


def depth_curve(group=P6):
    """Prefill rate, decode rate, acceptance and retrieval vs context depth."""
    for tag, label in group:
        d = load(os.path.join(R, f"{tag}-depth.json"))
        rows = (d or {}).get("depths") or []
        if not rows:
            continue
        needles = {n["target_tokens"]: n for n in (d.get("needle") or [])}
        print(f"\n**{label}**\n")
        print("| context | prompt tok | TTFT | prefill tok/s | decode tok/s | accept | needle |")
        print("|---:|---:|---:|---:|---:|---:|:--:|")
        for r in rows:
            if "error" in r:
                print(f"| {r['target_tokens']} | {r['prompt_tokens']} | "
                      f"error: {r['error'][:40]} | | | | |")
                continue
            n = needles.get(r["target_tokens"])
            found = "—" if not n else ("yes" if n["found"] else "**no**")
            k = r["prompt_tokens"]
            ctx = f"{k/1024:.0f}K" if k >= 1024 else str(k)
            print(f"| {ctx} | {k} | {r['ttft_s']:.2f} s | {r['prefill_tok_s']:.0f} | "
                  f"{r['decode_tok_s']:.1f} | {_acc(r.get('spec'))} | {found} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Q-bf16", help="tag used as the fidelity reference")
    a = ap.parse_args()
    speed_table()
    quality_table(a.ref)
    ladders()
    for name, group in (("Phase 6 (0.26.1)", P6), ("Phase 7 (0.28.0 vs 0.26.1)", P7),
                        ("Phase 8 (YaRN 512K, NGC)", P8)):
        if not any(load(os.path.join(R, f"{t}-probes.json")) or
                   load(os.path.join(R, f"{t}-ladder.json")) or
                   load(os.path.join(R, f"{t}-depth.json")) for t, _ in group):
            continue
        print(f"\n\n## {name}")
        per_probe_acceptance(group)
        concurrency_by_probe(group)
        depth_curve(group)
