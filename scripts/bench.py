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
import argparse, json, re, time, urllib.request, statistics, sys

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
    with urllib.request.urlopen(req, body, timeout=HTTP_TIMEOUT) as resp:
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

SPEC_COUNTERS = ("vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_emitted_tokens_total",
                 "vllm:spec_decode_num_drafts_total")
SPEC_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

def spec_counters(port):
    """Raw cumulative spec-decode counters ({} when spec decode is off).

    These are process-lifetime totals, so a single read describes every request the
    server has ever served. Diff two reads (spec_delta) to attribute acceptance to
    one probe instead of to the whole session.

    Includes the per-position vector: for a block drafter, *where* in the block
    acceptance collapses is the whole question, and the scalar rate hides it.
    """
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    vals = {}
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        for key in SPEC_COUNTERS:
            if ln.startswith(key + " ") or ln.startswith(key + "{"):
                vals[key] = float(ln.rsplit(" ", 1)[1])
        if ln.startswith(SPEC_PER_POS + "{"):
            m = re.search(r'position="(\d+)"', ln)
            if m:
                vals[f"{SPEC_PER_POS}:{m.group(1)}"] = float(ln.rsplit(" ", 1)[1])
    return vals

def spec_delta(before, after, k=None):
    """Acceptance over the window between two counter snapshots.

    accepted/drafted is the per-draft-token acceptance rate. num_drafts_total counts
    verification steps directly, so tokens emitted per target forward pass is
    accepted/drafts + 1 -- the quantity that actually sets decode speed -- without
    having to assume every step offered a full block. per_pos[i] / drafts is the
    probability that the i-th token of a draft block survives.
    """
    d = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in SPEC_COUNTERS}
    acc = d["vllm:spec_decode_num_accepted_tokens_total"]
    draft = d["vllm:spec_decode_num_draft_tokens_total"]
    steps = d["vllm:spec_decode_num_drafts_total"]
    if not draft:
        return None
    out = {"accepted": int(acc), "drafted": int(draft),
           "accepted_over_draft": round(acc / draft, 4)}
    if steps:
        out["steps"] = int(steps)
        out["tokens_per_target_pass"] = round(acc / steps + 1, 3)
        out["draft_tokens_per_step"] = round(draft / steps, 2)
        pos = []
        for i in range(int(round(draft / steps)) or (k or 0)):
            key = f"{SPEC_PER_POS}:{i}"
            if key not in after:
                break
            pos.append(round((after[key] - before.get(key, 0.0)) / steps, 4))
        if pos:
            out["per_position_acceptance"] = pos
    return out

CORPUS = "hf-cache/bench-corpus/mobydick.txt"
# Moby-Dick is ~340K tokens, too short for a 512K (YaRN) depth. Beyond its end the
# corpus continues with War and Peace (~900K tokens). Moby-Dick stays first, so every
# prompt up to ~330K tokens is byte-identical to the ones earlier phases measured.
CORPUS2 = "hf-cache/bench-corpus/warandpeace.txt"
CORPUS2_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
# Socket timeout for chat requests. vLLM streams nothing until the first token, so this
# has to cover the whole prefill: ~150 s at 128K, but an estimated ~20+ min at 512K.
HTTP_TIMEOUT = 900
# Deliberately real, non-repetitive prose. Tiled or generated filler is far more
# predictable than natural text, which would inflate drafter acceptance and make
# the depth curve measure the filler rather than the model.
NEEDLE = ("The harbormaster of Nantucket recorded the lantern count for that season "
          "as exactly 4817 units.")
NEEDLE_Q = ("What lantern count did the harbormaster of Nantucket record for that season? "
            "Reply with the number alone.")
NEEDLE_A = "4817"
# The needle question is answered in ~5 tokens, which is no decode phase at all. Decode
# rate at depth needs a prompt that actually generates, so the timed pass asks for a
# summary and the retrieval check keeps the short question.
DEPTH_Q = ("Summarise the events and themes of this excerpt in roughly 300 words, "
           "in flowing prose.")

CORPUS_URL = "https://www.gutenberg.org/files/2701/2701-0.txt"

def _gutenberg(rel, url):
    """Fetch a Project Gutenberg text on first use; return it without the license footer."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[corpus] fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, encoding="utf-8", errors="ignore") as f:
        t = f.read()
    end = t.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    return t[:end] if end > 0 else t

def corpus_text():
    """Long-context filler: Moby-Dick, then War and Peace, from Project Gutenberg.

    Both live under the git-ignored hf-cache/, so a clean clone pulls them rather than
    carrying megabytes of public-domain text in the repo.
    """
    t = _gutenberg(CORPUS, CORPUS_URL)
    # Skip the front matter *and* the table of contents: a chapter list is exactly the
    # repetitive text this probe is meant to avoid, and it would dominate a 1K-token depth.
    start = t.find("Call me Ishmael")
    t = t[start:] if start > 0 else t
    w = _gutenberg(CORPUS2, CORPUS2_URL)
    # War and Peace's contents list is just as repetitive; start at the first chapter text.
    s2 = w.find("Well, Prince, so Genoa and Lucca")
    w = w[s2:] if s2 > 0 else w
    return t.rstrip() + "\n\n" + w

def count_tokens(port, text):
    """Exact prompt length from the server's own tokenizer (falls back to chars/3.6)."""
    try:
        r = http_json(f"http://127.0.0.1:{port}/tokenize",
                      {"model": "qwen3.8-27b", "prompt": text}, timeout=120)
        return int(r["count"])
    except Exception:
        return int(len(text) / 3.6)

def build_depth_prompt(port, target_tokens, text, question=None, nonce=""):
    """Corpus excerpt sized to ~target_tokens, with the needle buried at ~60% depth."""
    chars = int(target_tokens * 3.6)
    for _ in range(6):                      # converge on the token target, then stop
        body = text[:max(chars, 200)]
        n = count_tokens(port, body)
        if abs(n - target_tokens) <= max(64, target_tokens * 0.02):
            break
        chars = int(chars * target_tokens / max(n, 1))
    cut = int(len(body) * 0.6)
    cut = body.find(". ", cut)
    cut = cut + 2 if cut > 0 else int(len(body) * 0.6)
    body = body[:cut] + " " + NEEDLE + " " + body[cut:]
    # The nonce leads, so a cached prefix cannot match any earlier prompt.
    prompt = (nonce + "Read the following excerpt, then answer the question at the end.\n\n"
              + body + "\n\n" + (question or NEEDLE_Q))
    return prompt, count_tokens(port, prompt)

def depth_sweep(port, depths, max_tokens=300, temperature=0.7, top_p=0.8, spec_tokens=None,
                warm=True):
    """Decode rate, prefill rate and drafter acceptance as a function of context depth.

    Every other number in this repo comes from a ~100-token prompt, so nothing in it
    says what happens at the context length the recipe advertises. Each depth also
    carries a needle, so the row reports whether the model still reads what it holds.

    Prefix caching is ON in upstream vLLM for this architecture, so every prompt is
    prefixed with a unique nonce -- without it the second measurement of a given depth
    reads back a cached prefill and reports an impossible prefill rate. The warm repeat
    is then measured on purpose, since that is what a real multi-turn client sees.
    """
    text = corpus_text()
    out = []
    for d in depths:
        nonce = f"[run {time.time_ns():x}] "
        prompt, actual = build_depth_prompt(port, d, text, DEPTH_Q, nonce)
        c0 = spec_counters(port)
        try:
            r = stream_chat(port, prompt, max_tokens, temperature, top_p)
        except Exception as e:
            out.append({"target_tokens": d, "prompt_tokens": actual, "error": str(e)[:200]})
            print("[depth]", out[-1]); continue
        row = {
            "target_tokens": d,
            "prompt_tokens": actual,
            "ttft_s": r["ttft_s"],
            "prefill_tok_s": round(actual / max(r["ttft_s"], 1e-6), 1),
            "decode_tok_s": r["net_decode_tok_s"],
            "completion_tokens": r["completion_tokens"],
            "spec": spec_delta(c0, spec_counters(port), spec_tokens),
        }
        # Same prompt again: everything but the generation is a prefix-cache hit.
        # Skippable: without prefix caching (NGC) it is a second full prefill, which
        # at 512K is ~20 minutes spent measuring nothing.
        if not warm:
            out.append(row); print("[depth]", row); continue
        try:
            w = stream_chat(port, prompt, 16, temperature, top_p)
            row["ttft_warm_s"] = w["ttft_s"]
            row["prefill_speedup_warm"] = round(r["ttft_s"] / max(w["ttft_s"], 1e-6), 1)
        except Exception as e:
            row["ttft_warm_s"] = None
            row["warm_error"] = str(e)[:120]
        out.append(row)
        print("[depth]", row)
    return out

def depth_needle_check(port, depths, temperature=0.0, top_p=1.0):
    """Separate greedy pass for retrieval correctness; kept out of the timed rows."""
    text = corpus_text()
    out = []
    for d in depths:
        prompt, actual = build_depth_prompt(port, d, text, NEEDLE_Q,
                                            f"[needle {time.time_ns():x}] ")
        try:
            r = stream_chat_text(port, prompt, 24, temperature, top_p)
            ok = NEEDLE_A in r
        except Exception as e:
            r, ok = str(e)[:120], False
        out.append({"target_tokens": d, "prompt_tokens": actual,
                    "found": ok, "answer": r.strip()[:80]})
        print("[needle]", out[-1])
    return out

def stream_chat_text(port, prompt, max_tokens, temperature=0.7, top_p=0.8):
    """Non-streamed completion returning just the text (correctness checks, not timing)."""
    r = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=HTTP_TIMEOUT)
    return r["choices"][0]["message"].get("content") or ""

def metrics_spec_acceptance(port):
    """MTP acceptance rate from Prometheus counters (None if spec decode absent)."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    vals = {}
    for key in SPEC_COUNTERS:
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

def concurrency_ladder(port, probe, streams_list, max_tokens, temperature=0.7, top_p=0.8,
                       spec_tokens=None):
    """Fire N simultaneous requests; report aggregate + per-stream tok/s and TTFT."""
    import concurrent.futures as cf
    out = []
    prompt = PROBES[probe]["prompt"]
    for n in streams_list:
        c0 = spec_counters(port)
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
        # A drafter that degrades under load shows up here before it shows up in tok/s.
        row["spec"] = spec_delta(c0, spec_counters(port), spec_tokens)
        row["probe"] = probe
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
    ap.add_argument("--ladder-probes", default="essay",
                    help="comma list of probes to run the ladder on (default essay only)")
    ap.add_argument("--depths", default="",
                    help="comma list of context depths in tokens, e.g. 1024,16384,131072")
    ap.add_argument("--needle", action="store_true",
                    help="with --depths, also run a greedy retrieval check at each depth")
    ap.add_argument("--no-warm", action="store_true",
                    help="with --depths, skip the warm (prefix-cache) repeat of each prompt")
    ap.add_argument("--http-timeout", type=int, default=900,
                    help="socket timeout (s) for chat requests; must exceed the longest prefill")
    ap.add_argument("--spec-tokens", type=int, default=None,
                    help="num_speculative_tokens the server was launched with; enables "
                         "tokens-per-target-pass in the per-probe spec breakdown")
    args = ap.parse_args()
    global HTTP_TIMEOUT
    HTTP_TIMEOUT = args.http_timeout

    if not wait_server(args.port):
        print("server never became ready", file=sys.stderr); sys.exit(1)

    # One short warm-up per probe, so the first timed run never pays cold-path cost.
    # Deliberately NOT the probe's own prompt: prefix caching is off for this hybrid
    # arch, so a warm-up on the real prompt would buy nothing and only add drift.
    for _ in range(args.warmup * len(args.probes.split(","))):
        stream_chat(args.port, "Warm up with one short sentence about GPUs.", 32,
                    thinking=args.thinking)

    results = {"tag": args.tag, "port": args.port, "timestamp": time.strftime("%F %T"),
               "sampling": {"temperature": args.temperature, "top_p": args.top_p}, "probes": {}}
    for p in args.probes.split(","):
        runs = []
        # Counter snapshot around this probe only: the aggregate `mtp` block below
        # cannot separate probes, and whether acceptance is content-dependent is
        # exactly the question a block drafter raises.
        c0 = spec_counters(args.port)
        for _ in range(args.n):
            kw = {}
            if args.temperature is not None:
                kw["temperature"] = args.temperature
            if args.top_p is not None:
                kw["top_p"] = args.top_p
            runs.append(stream_chat(args.port, PROBES[p]["prompt"], PROBES[p]["max_tokens"],
                                    thinking=args.thinking, **kw))
        spec = spec_delta(c0, spec_counters(args.port), args.spec_tokens)
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        results["probes"][p] = {"median": med, "runs": runs, "spec": spec}
        print(f"[{p}] net={med['net_decode_tok_s']} tok/s  vis={med['visible_decode_tok_s']}  "
              f"ttft={med['ttft_s']}s  e2e={med['e2e_tok_s']} tok/s  ({args.n} reps)")
        if spec:
            extra = (f"  tok/target-pass={spec['tokens_per_target_pass']}"
                     if "tokens_per_target_pass" in spec else "")
            print(f"[{p}] accept={spec['accepted_over_draft']} "
                  f"({spec['accepted']}/{spec['drafted']}){extra}")

    m = metrics_spec_acceptance(args.port)
    results["mtp"] = m
    print(f"[mtp] acceptance={m}" if m else "[mtp] no spec-decode counters found")

    if args.ladder:
        temps = args.temperature if args.temperature is not None else 0.7
        tps = args.top_p if args.top_p is not None else 0.8
        streams = [int(x) for x in args.ladder.split(",")]
        results["ladders"] = {}
        for lp in args.ladder_probes.split(","):
            print(f"--- ladder on {lp} ---")
            results["ladders"][lp] = concurrency_ladder(
                args.port, lp, streams, PROBES[lp]["max_tokens"], temps, tps,
                spec_tokens=args.spec_tokens)
        # `ladder` stays the essay rung list: summarize.py and every earlier result
        # file use that key. Falls back to the first probe laddered.
        results["ladder"] = results["ladders"].get(
            "essay", next(iter(results["ladders"].values())))

    if args.depths:
        temps = args.temperature if args.temperature is not None else 0.7
        tps = args.top_p if args.top_p is not None else 0.8
        ds = [int(x) for x in args.depths.split(",")]
        results["depths"] = depth_sweep(args.port, ds, temperature=temps, top_p=tps,
                                        spec_tokens=args.spec_tokens,
                                        warm=not args.no_warm)
        if args.needle:
            results["needle"] = depth_needle_check(args.port, ds)

    out = args.out or f"results/{args.tag}.json"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print("saved ->", out)

if __name__ == "__main__":
    main()
