#!/usr/bin/env python3
"""Quality gate: fixed 12-question exact-match sanity check (temperature 0, thinking off).

Catches quantization breakage (garbage output, arithmetic failure) cheaply.
Usage: python3 quality.py --port 8000 [--out results/quality-X.json]
"""
import argparse, json, time, urllib.request

QUESTIONS = [
    ("What is 17 * 23?", "391"),
    ("What is 1024 + 640?", "1664"),
    ("What is the capital of Australia? Answer with the city name only.", "Canberra"),
    ("How many strings does a standard guitar have? One number only.", "6"),
    ("What is the square root of 144?", "12"),
    ("If a train travels 60 km in 45 minutes, what is its speed in km/h? Number only.", "80"),
    ("What year did the Berlin Wall fall? Year only.", "1989"),
    ("Name the largest planet in our solar system. One word.", "Jupiter"),
    ("What is 15% of 200? Number only.", "30"),
    ("In Python, what does len([1,2,3]) return? Number only.", "3"),
    ("What is the next prime after 13? Number only.", "17"),
    ("Spell the chemical symbol for gold. Two letters.", "Au"),
]

def ask(port, q):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 256,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, json.dumps(payload).encode(), timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    score = 0
    details = []
    for q, want in QUESTIONS:
        try:
            ans = ask(args.port, q)
            ok = want.lower() in ans.lower()
        except Exception as e:
            ans, ok = f"ERROR {e}", False
        score += ok
        details.append({"q": q, "want": want, "got": ans[:120], "ok": ok})
        print(("PASS" if ok else "FAIL"), "|", q[:50], "->", ans[:60].replace("\n", " "))
    print(f"\nSCORE: {score}/{len(QUESTIONS)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"score": score, "total": len(QUESTIONS), "details": details}, f, indent=2)

if __name__ == "__main__":
    main()
