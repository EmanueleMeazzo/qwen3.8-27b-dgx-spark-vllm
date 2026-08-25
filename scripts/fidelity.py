#!/usr/bin/env python3
"""Quantization-fidelity probe: how far has this checkpoint's next-token
distribution drifted from a reference checkpoint's?

Why this exists: the 12-question sanity gate saturates -- every quant scores 12/12 --
so it cannot rank checkpoints. This measures the thing quantization actually damages,
the next-token distribution itself, on fixed text, with no sampling involved.

Why it is built the way it is: the obvious implementation (`echo=true` + `logprobs`,
i.e. prompt logprobs) is SILENTLY WRONG on this model. On the hybrid GDN architecture
the NGC 26.07 engine returns garbage for non-final prompt positions -- a token the
decode path scores at -1.04 comes back as -17.13 with junk alternatives. So every
probe here goes through the normal decode path: feed the true prefix, read the
next-token distribution for exactly one position.

Metrics, per probe position:
  * nll        - -logprob of the token that actually follows, when it is inside the
                 returned top-k (coverage is reported; a checkpoint that pushes the
                 true token out of its top-20 is itself a quality signal).
  * top1       - the checkpoint's argmax token, for agreement against a reference.
  * kl         - KL(reference || this) over the reference's top-k support, which is
                 the honest "how different is this distribution" number.

Usage:
  python3 fidelity.py --port 8000 --tag bf16   --out results/fid-bf16.json
  python3 fidelity.py --port 8000 --tag int4   --out results/fid-int4.json \
                      --ref results/fid-bf16.json
"""
import argparse, json, math, os, urllib.request

PASSAGES = {
    "prose": (
        "The unified memory architecture of the GB10 superchip removes the explicit copy "
        "between host and device that has shaped accelerator programming for two decades. "
        "Because the CPU and GPU address the same physical LPDDR5X, a tensor produced by "
        "the tokenizer is visible to the attention kernel without a transfer, and the "
        "practical limit on model size becomes the total memory of the box rather than the "
        "capacity of a discrete card. The tradeoff is bandwidth: a soldered LPDDR5X pool "
        "delivers a fraction of what stacked HBM provides, so decode throughput for a dense "
        "model is set almost entirely by how many bytes of weights must be read per token."
    ),
    "code": (
        "import functools\n"
        "from collections import OrderedDict\n\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int) -> None:\n"
        "        if capacity <= 0:\n"
        "            raise ValueError('capacity must be positive')\n"
        "        self.capacity = capacity\n"
        "        self._data: OrderedDict[int, int] = OrderedDict()\n\n"
        "    def get(self, key: int) -> int:\n"
        "        if key not in self._data:\n"
        "            return -1\n"
        "        self._data.move_to_end(key)\n"
        "        return self._data[key]\n\n"
        "    def put(self, key: int, value: int) -> None:\n"
        "        if key in self._data:\n"
        "            self._data.move_to_end(key)\n"
        "        self._data[key] = value\n"
        "        if len(self._data) > self.capacity:\n"
        "            self._data.popitem(last=False)\n"
    ),
    "math": (
        "Let f(x) = x^3 - 6x^2 + 9x + 1. To find the local extrema we differentiate: "
        "f'(x) = 3x^2 - 12x + 9 = 3(x^2 - 4x + 3) = 3(x - 1)(x - 3). The critical points are "
        "therefore x = 1 and x = 3. The second derivative is f''(x) = 6x - 12, so f''(1) = -6 "
        "is negative and x = 1 is a local maximum, while f''(3) = 6 is positive and x = 3 is a "
        "local minimum. Evaluating, f(1) = 1 - 6 + 9 + 1 = 5 and f(3) = 27 - 54 + 27 + 1 = 1."
    ),
    "chinese": (
        "统一内存架构的核心优势在于消除了主机与设备之间的显式数据拷贝。由于中央处理器和图形处理器"
        "访问同一块物理内存，分词器产生的张量无需传输即可被注意力核函数读取，因此模型规模的实际"
        "上限取决于整机内存容量，而不再取决于独立显卡的显存。代价是带宽：焊装的低功耗内存所提供的"
        "带宽远低于堆叠式高带宽内存，所以稠密模型的解码吞吐几乎完全由每个词元需要读取的权重字节数决定。"
    ),
}


def post(port, path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, json.dumps(payload).encode(), timeout=timeout) as r:
        return json.loads(r.read())


def token_strings(port, model, text):
    """Per-token surface strings. Only the `tokens` list is used -- the logprobs that
    come back on this path are the broken ones described in the module docstring."""
    d = post(port, "/v1/completions", {
        "model": model, "prompt": text, "max_tokens": 1,
        "temperature": 0.0, "echo": True, "logprobs": 0,
    })
    return d["choices"][0]["logprobs"]["tokens"]


def probe(port, model, ids_prefix_text, top_k):
    """Next-token distribution after a prefix, via the normal decode path."""
    d = post(port, "/v1/completions", {
        "model": model, "prompt": ids_prefix_text, "max_tokens": 1,
        "temperature": 0.0, "logprobs": top_k,
    })
    return d["choices"][0]["logprobs"]["top_logprobs"][0]


def kl_ref_vs(ref_dist, cur_dist):
    """KL(ref || cur), both restricted to the reference's top-k support and each
    renormalized over exactly that support, so identical distributions score 0.

    A token the candidate does not rank at all is floored just below its weakest
    returned logprob -- that is the penalty for having dropped it out of the top-k.
    """
    if not ref_dist:
        return 0.0
    floor = (min(cur_dist.values()) - 2.0) if cur_dist else -25.0
    ref_lps = {t: lp for t, lp in ref_dist.items()}
    cur_lps = {t: cur_dist.get(t, floor) for t in ref_lps}
    zr = sum(math.exp(v) for v in ref_lps.values())
    zc = sum(math.exp(v) for v in cur_lps.values())
    kl = 0.0
    for t, lp in ref_lps.items():
        p = math.exp(lp) / zr
        if p <= 0.0:
            continue
        kl += p * ((lp - math.log(zr)) - (cur_lps[t] - math.log(zc)))
    return max(kl, 0.0)


def compare(ref_path, cand_paths):
    """Offline: rank saved probe files against a reference, no server needed.

    Each checkpoint is probed once and its distributions saved; every pairwise
    comparison is then just arithmetic, so the reference does not have to exist yet
    when the candidates are measured.
    """
    ref = json.load(open(ref_path))
    print(f"reference: {ref.get('tag')}  (ppl geomean {ref.get('ppl_geomean')})\n")
    hdr = f"{'checkpoint':28s} {'ppl':>8s} {'ppl/ref':>8s} {'top1_agree':>11s} {'KL(ref||x)':>11s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for cp in cand_paths:
        cur = json.load(open(cp))
        agree = n = 0
        kls = []
        for name, rec in cur["passages"].items():
            rr = ref["passages"].get(name)
            if not rr:
                continue
            for a, b in zip(rec["probes"], rr["probes"]):
                if a["pos"] != b["pos"] or a["target"] != b["target"]:
                    continue  # tokenizers disagree here; not comparable
                n += 1
                agree += a["top1"] == b["top1"]
                kls.append(kl_ref_vs(b["dist"], a["dist"]))
        row = {
            "tag": cur.get("tag"),
            "ppl_geomean": cur.get("ppl_geomean"),
            "ppl_ratio": (round(cur["ppl_geomean"] / ref["ppl_geomean"], 4)
                          if cur.get("ppl_geomean") and ref.get("ppl_geomean") else None),
            "top1_agree": round(agree / max(n, 1), 4),
            "kl_from_ref": round(sum(kls) / max(len(kls), 1), 5),
            "n_compared": n,
        }
        rows.append(row)
        print(f"{row['tag']:28s} {row['ppl_geomean']:8.4f} {row['ppl_ratio']:8.4f} "
              f"{row['top1_agree']:11.4f} {row['kl_from_ref']:11.5f}")
    return {"ref": ref.get("tag"), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", default=None,
                    metavar=("REF.json", "CAND.json"),
                    help="offline mode: rank saved probe files against the first one")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8, help="probe every Nth token")
    ap.add_argument("--start", type=int, default=16, help="first probed position")
    args = ap.parse_args()

    if args.compare:
        res = compare(args.compare[0], args.compare[1:])
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(res, open(args.out, "w"), indent=2)
            print("\nsaved ->", args.out)
        return

    out = {"tag": args.tag, "top_k": args.top_k, "stride": args.stride, "passages": {}}
    for name, text in PASSAGES.items():
        toks = token_strings(args.port, args.model, text)
        # drop the single generated token at the tail; it is not part of the passage
        toks = toks[:-1]
        positions = list(range(args.start, len(toks), args.stride))
        rec = {"n_probes": len(positions), "probes": []}
        for pos in positions:
            prefix = "".join(toks[:pos])
            dist = probe(args.port, args.model, prefix, args.top_k)
            target = toks[pos]
            top1 = max(dist.items(), key=lambda kv: kv[1])[0]
            rec["probes"].append({
                "pos": pos,
                "target": target,
                "target_lp": dist.get(target),
                "top1": top1,
                "dist": {k: round(v, 5) for k, v in dist.items()},
            })
        covered = [p for p in rec["probes"] if p["target_lp"] is not None]
        rec["coverage"] = round(len(covered) / max(len(rec["probes"]), 1), 4)
        if covered:
            nll = -sum(p["target_lp"] for p in covered) / len(covered)
            rec["ppl_covered"] = round(math.exp(nll), 4)
        out["passages"][name] = rec
        print(f"[{name:8s}] probes={rec['n_probes']:3d} coverage={rec['coverage']:.3f} "
              f"ppl(covered)={rec.get('ppl_covered')}")

    ppls = [v["ppl_covered"] for v in out["passages"].values() if v.get("ppl_covered")]
    if ppls:
        out["ppl_geomean"] = round(math.exp(sum(math.log(p) for p in ppls) / len(ppls)), 4)
        print(f"\nPPL geomean (covered positions): {out['ppl_geomean']}")

    if args.ref:
        ref = json.load(open(args.ref))
        agree = n = 0
        kls = []
        for name, rec in out["passages"].items():
            rr = ref["passages"].get(name)
            if not rr:
                continue
            pa = pn = 0
            pk = []
            for cp, rp in zip(rec["probes"], rr["probes"]):
                if cp["pos"] != rp["pos"]:
                    continue
                pn += 1
                pa += cp["top1"] == rp["top1"]
                pk.append(kl_ref_vs(rp["dist"], cp["dist"]))
            rec["top1_agree"] = round(pa / max(pn, 1), 4)
            rec["kl_from_ref"] = round(sum(pk) / max(len(pk), 1), 5)
            agree += pa; n += pn; kls.extend(pk)
            print(f"[{name:8s}] top1_agree={rec['top1_agree']:.4f}  "
                  f"KL(ref||this)={rec['kl_from_ref']:.5f}")
        out["vs_ref"] = {
            "ref_tag": ref.get("tag"),
            "top1_agree": round(agree / max(n, 1), 4),
            "kl_from_ref": round(sum(kls) / max(len(kls), 1), 5),
            "ppl_ratio": (round(out["ppl_geomean"] / ref["ppl_geomean"], 4)
                          if ppls and ref.get("ppl_geomean") else None),
        }
        v = out["vs_ref"]
        print(f"\nOVERALL vs {v['ref_tag']}: top1_agree={v['top1_agree']}  "
              f"KL={v['kl_from_ref']}  ppl_ratio={v['ppl_ratio']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
