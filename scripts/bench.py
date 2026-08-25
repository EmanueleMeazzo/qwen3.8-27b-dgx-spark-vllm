#!/usr/bin/env python3
"""Benchmark client for Qwen3.8-27B served by vLLM on DGX Spark.

Measures, per probe:
  - TTFT (time to first token)
  - net decode tok/s (post-first-token, from server usage completion_tokens)
  - end-to-end tok/s incl. prefill
  - MTP acceptance rate (from /metrics spec-decode counters)

Probes mirror the SGLang reference recipe so numbers are comparable in spirit:
  code  : implement an LRU cache (agentic/code style)
  essay : long-form writing (Babbage->GPUs)
  chat  : short conversational answer

Usage:
  python3 bench.py --port 8000 --n 3 --probes code,essay --out results/foo.json
"""
import argparse, json, time, urllib.request, statistics, sys

PROBES = {
    "code": {
        "prompt": "Write a Python class LRUCache with get(key) and put(key,value) in O(1) using "
                  "an ordered dict. Then write five unit tests with pytest and explain the eviction "
                  "order. Include the full implementation.",
        "max_tokens": 700,
    },
    "essay": {
        "prompt": "Write a well-structured essay of about 500 words on how computing machines evolved "
                  "from Charles Babbage's Analytical Engine to modern GPUs, ending with what unified "
                  "memory means for the future of AI hardware.",
        "max_tokens": 700,
    },
    "chat": {
        "prompt": "What is a hash map, why is average lookup O(1), and when would you use a balanced "
                  "tree instead? Answer concisely in a few sentences.",
        "max_tokens": 300,
    },
}

def http_json(url, payload=None, timeout=600):
    req = urllib.request.Request(url)
    data = None
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, data, timeout=timeout) as r:
        return json.loads(r.read())

def wait_server(port, timeout_s=3600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            http_json(f"http://127.0.0.1:{port}/v1/models")
            return True
        except Exception:
            time.sleep(5)
    return False

def stream_chat(port, prompt, max_tokens, temperature=0.7, top_p=0.8, thinking=False):
    """Streamed completion; returns dict with timing + token counts."""
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions")
    req.add_header("Content-Type", "application/json")
    body = json.dumps(payload).encode()

    t_start = time.perf_counter()
    ttft = None
    completion_tokens = None
    reasoning_tokens = 0
    with urllib.request.urlopen(req, body, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            u = chunk.get("usage")
            if u:
                completion_tokens = u["completion_tokens"]
                details = u.get("completion_tokens_details") or {}
                reasoning_tokens = details.get("reasoning_tokens", 0) or 0
                continue
            if chunk.get("choices") and ttft is None:
                ttft = time.perf_counter() - t_start
    t_end = time.perf_counter()
    if ttft is None or not completion_tokens:
        raise RuntimeError("no tokens received / no usage chunk")
    decode_time = t_end - t_start - ttft
    visible = max(1, completion_tokens - reasoning_tokens)
    return {
        "ttft_s": round(ttft, 3),
        "total_s": round(t_end - t_start, 3),
        "decode_s": round(decode_time, 3),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        # decode speed over ALL generated tokens (thinking included) — matches vLLM's own accounting
        "net_decode_tok_s": round(completion_tokens / decode_time, 2),
        "visible_decode_tok_s": round(visible / decode_time, 2),
        "e2e_tok_s": round(completion_tokens / (t_end - t_start), 2),
    }

def metrics_spec_acceptance(port):
    """MTP acceptance rate from Prometheus counters (None if spec decode absent)."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    vals = {}
    for key in ("vllm:spec_decode_num_accepted_tokens_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_emitted_tokens_total"):
        for ln in text.splitlines():
            if ln.startswith(key + " ") or ln.startswith(key + "{"):
                vals[key] = float(ln.rsplit(" ", 1)[1])
    acc = vals.get("vllm:spec_decode_num_accepted_tokens_total")
    draft = vals.get("vllm:spec_decode_num_draft_tokens_total")
    out = {}
    if acc is not None and draft:
        out["accepted_over_draft"] = round(acc / draft, 4)
        out["accepted"], out["drafted"] = int(acc), int(draft)
    # engine-computed mean accepted tokens per decode step (the honest metric)
    for ln in text.splitlines():
        if ln.startswith("vllm:spec_decode_mean_acceptance_length"):
            try:
                out["mean_acceptance_length"] = round(float(ln.rsplit(" ", 1)[1]), 3)
            except ValueError:
                pass
    return out or None

def concurrency_ladder(port, probe, streams_list, max_tokens, temperature=0.7, top_p=0.8):
    """Fire N simultaneous requests; report aggregate + per-stream tok/s and TTFT."""
    import concurrent.futures as cf
    out = []
    prompt = PROBES[probe]["prompt"]
    for n in streams_list:
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            t0 = time.perf_counter()
            futs = [ex.submit(stream_chat, port, prompt, max_tokens, temperature, top_p) for _ in range(n)]
            results_list = []
            errors = 0
            for f in futs:
                try:
                    results_list.append(f.result())
                except Exception:
                    errors += 1
        wall = time.perf_counter() - t0
        toks = sum(r["completion_tokens"] for r in results_list)
        if not results_list:
            row = {"streams": n, "errors": errors, "aggregate_tok_s": 0,
                   "per_stream_tok_s": 0, "ttft_med_s": None, "ttft_max_s": None}
        else:
            row = {
                "streams": n,
                "errors": errors,
                "wall_s": round(wall, 2),
                "aggregate_tok_s": round(toks / max(wall, 0.001), 1),
                "per_stream_tok_s": round(sum(r["net_decode_tok_s"] for r in results_list) / len(results_list), 1),
                "ttft_med_s": round(statistics.median(r["ttft_s"] for r in results_list), 3),
                "ttft_max_s": round(max(r["ttft_s"] for r in results_list), 3),
            }
        out.append(row)
        print("[ladder]", row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--probes", default="code,essay,chat")
    ap.add_argument("--n", type=int, default=3, help="repeats per probe (median reported)")
    ap.add_argument("--thinking", action="store_true", help="enable thinking mode (default off for bench stability)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--temperature", type=float, default=None, help="override probe temperature")
    ap.add_argument("--top-p", type=float, default=None, help="override probe top_p")
    ap.add_argument("--ladder", default="", help="comma list of stream counts, e.g. 1,2,4,8 (runs after probes)")
    args = ap.parse_args()

    if not wait_server(args.port):
        print("server never became ready", file=sys.stderr); sys.exit(1)

    for p in args.probes.split(","):
        pr = PROBES[p]
        stream_chat(args.port, "Warm up with one short sentence about GPUs.", 32, thinking=args.thinking) if args.warmup else None

    results = {"tag": args.tag, "port": args.port, "timestamp": time.strftime("%F %T"),
               "sampling": {"temperature": args.temperature, "top_p": args.top_p}, "probes": {}}
    for p in args.probes.split(","):
        runs = []
        for _ in range(args.n):
            kw = {}
            if args.temperature is not None:
                kw["temperature"] = args.temperature
            if args.top_p is not None:
                kw["top_p"] = args.top_p
            runs.append(stream_chat(args.port, PROBES[p]["prompt"], PROBES[p]["max_tokens"],
                                    thinking=args.thinking, **kw))
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        results["probes"][p] = {"median": med, "runs": runs}
        print(f"[{p}] net={med['net_decode_tok_s']} tok/s  vis={med['visible_decode_tok_s']}  "
              f"ttft={med['ttft_s']}s  e2e={med['e2e_tok_s']} tok/s  ({args.n} reps)")

    m = metrics_spec_acceptance(args.port)
    results["mtp"] = m
    print(f"[mtp] acceptance={m}" if m else "[mtp] no spec-decode counters found")

    if args.ladder:
        temps = args.temperature if args.temperature is not None else 0.7
        tps = args.top_p if args.top_p is not None else 0.8
        results["ladder"] = concurrency_ladder(
            args.port, "essay", [int(x) for x in args.ladder.split(",")],
            PROBES["essay"]["max_tokens"], temps, tps)

    out = args.out or f"results/{args.tag}.json"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print("saved ->", out)

if __name__ == "__main__":
    main()
