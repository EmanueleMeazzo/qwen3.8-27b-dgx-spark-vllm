#!/usr/bin/env python3
"""Capability gate: multi-step reasoning + executed code.

The original quality.py asks twelve one-hop facts ("what is 17*23"), which every
checkpoint here answers correctly -- it is a breakage detector, not a ranking. This
one is built to separate checkpoints that are merely *working* from checkpoints that
are still *good*: every item needs several dependent steps, so a distribution that
has drifted slightly derails somewhere along the chain.

Two sections:
  reasoning  - 24 multi-step word problems, greedy, answer checked against an exact
               expected value parsed out of the tail of the response.
  code       - 8 function-writing tasks, verified by actually running the generated
               function against hidden asserts (a self-verifying signal that needs no
               reference model).

Usage: python3 capability.py --port 8000 --out results/cap-X.json
"""
import argparse, json, os, re, subprocess, sys, tempfile, urllib.request

REASONING = [
    ("A train leaves at 09:40 and the trip takes 3 hours 55 minutes. There is then a 25 minute "
     "delay. What time does it arrive? Answer in 24-hour HH:MM.", "14:00"),
    ("A shop sells pens at 3 for $4.50. I buy 14 pens. Individually they cost $1.80 each. "
     "How much do I save buying in packs where possible, paying single price for leftovers? "
     "Answer in dollars like $X.XX", "$3.60"),
    ("Start with 128. Halve it three times, then add 17, then multiply by 4. What is the result?", "132"),
    ("A rectangle has perimeter 46 cm and its length is 5 cm more than its width. What is its "
     "area in square centimetres? Number only.", "126"),
    ("If today is Wednesday, what day of the week is it 100 days from now?", "Friday"),
    ("A tank holds 240 litres. It is 5/8 full. I remove 45 litres, then add 30. What fraction of "
     "the tank is now full? Answer as a simplified fraction.", "9/16"),
    ("Three consecutive even integers sum to 138. What is the largest one? Number only.", "48"),
    ("I have 5 red, 3 blue and 2 green marbles. I draw two without replacement. What is the "
     "probability both are red? Answer as a simplified fraction.", "2/9"),
    ("A car travels 150 km at 75 km/h, then 90 km at 45 km/h. What is the average speed for the "
     "whole trip in km/h? Number only.", "60"),
    ("What is 15% of 15% of 8000? Number only.", "180"),
    ("A number increased by 30% is 260. What was the original number? Number only.", "200"),
    ("In a class of 30, 18 play football, 15 play tennis, and 7 play both. How many play neither? "
     "Number only.", "4"),
    ("The sequence goes 2, 6, 12, 20, 30, ... What is the 8th term? Number only.", "72"),
    ("A book costs $24 after a 20% discount. What was the original price in dollars? Number only.", "30"),
    ("If 3 machines make 3 widgets in 3 minutes, how long do 100 machines take to make 100 widgets? "
     "Answer in minutes, number only.", "3"),
    ("Solve for x: 2(x - 3) + 4 = 3x - 7. Number only.", "5"),
    ("A cube has surface area 150 cm^2. What is its volume in cubic centimetres? Number only.", "125"),
    ("I start with $100. I lose 20%, then gain 20% of the new amount. How much do I have? "
     "Answer in dollars like $XX.XX", "$96.00"),
    ("What is the sum of all integers from 1 to 100 that are divisible by 7? Number only.", "735"),
    ("A recipe for 4 people needs 300 g flour. I cook for 7 people. How many grams of flour, "
     "rounded to the nearest gram? Number only.", "525"),
    ("Two trains 300 km apart approach each other at 60 km/h and 90 km/h. After how many hours "
     "do they meet? Number only.", "2"),
    ("What is the remainder when 7^100 is divided by 5? Number only.", "1"),
    ("A right triangle has legs 9 and 12. What is the length of the hypotenuse? Number only.", "15"),
    ("If x + y = 12 and xy = 35, what is x^2 + y^2? Number only.", "74"),
]

CODE = [
    ("Write a Python function `run_length_encode(s)` that returns a list of (char, count) tuples "
     "for consecutive runs in the string s. Return only the function.",
     "run_length_encode",
     "assert run_length_encode('aaabbc') == [('a',3),('b',2),('c',1)]\n"
     "assert run_length_encode('') == []\n"
     "assert run_length_encode('x') == [('x',1)]"),
    ("Write a Python function `balanced(s)` that returns True if the brackets in s -- (), [] and {} "
     "-- are correctly nested and matched, ignoring all other characters. Return only the function.",
     "balanced",
     "assert balanced('a(b[c]{d})') is True\n"
     "assert balanced('([)]') is False\n"
     "assert balanced('') is True\n"
     "assert balanced('(') is False"),
    ("Write a Python function `merge_intervals(iv)` taking a list of [start, end] pairs and "
     "returning the merged, sorted list of non-overlapping intervals. Return only the function.",
     "merge_intervals",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([]) == []\n"
     "assert merge_intervals([[1,4],[3,5]]) == [[1,5]]"),
    ("Write a Python function `roman_to_int(s)` converting a Roman numeral string to an integer, "
     "handling subtractive pairs. Return only the function.",
     "roman_to_int",
     "assert roman_to_int('MCMXCIV') == 1994\n"
     "assert roman_to_int('III') == 3\n"
     "assert roman_to_int('LVIII') == 58"),
    ("Write a Python function `word_frequencies(text)` returning a dict mapping each lowercase word "
     "to its count. Words are maximal runs of letters and apostrophes. Return only the function.",
     "word_frequencies",
     "assert word_frequencies(\"The cat, the CAT!\") == {'the':2,'cat':2}\n"
     "assert word_frequencies('') == {}\n"
     "assert word_frequencies(\"don't don't\") == {\"don't\":2}"),
    ("Write a Python function `spiral(matrix)` returning the elements of a 2-D list in clockwise "
     "spiral order as a flat list. Return only the function.",
     "spiral",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n"
     "assert spiral([]) == []\n"
     "assert spiral([[1,2],[3,4]]) == [1,2,4,3]"),
    ("Write a Python function `longest_common_prefix(strs)` returning the longest common prefix of "
     "a list of strings, or '' if there is none. Return only the function.",
     "longest_common_prefix",
     "assert longest_common_prefix(['flower','flow','flight']) == 'fl'\n"
     "assert longest_common_prefix(['dog','racecar']) == ''\n"
     "assert longest_common_prefix([]) == ''"),
    ("Write a Python function `base_convert(n, b)` returning the string representation of the "
     "non-negative integer n in base b (2 <= b <= 16), using lowercase digits. Return only the function.",
     "base_convert",
     "assert base_convert(255, 16) == 'ff'\n"
     "assert base_convert(0, 2) == '0'\n"
     "assert base_convert(10, 2) == '1010'"),
]


def ask(port, model, q, max_tokens=1024):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, json.dumps(payload).encode(), timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def check_reasoning(ans, want):
    """The expected value must appear in the last stretch of the answer, so that a model
    that merely mentions the right number mid-derivation and then concludes wrongly
    does not score."""
    tail = ans.strip()[-160:]
    if want in tail:
        return True
    # tolerate 27/48 vs 9/16 style equivalent fractions and $4.20 vs 4.20
    alt = want.lstrip("$")
    if alt and alt in tail:
        return True
    if "/" in want:
        try:
            a, b = (int(x) for x in want.split("/"))
            from math import gcd
            g = gcd(a, b)
            if f"{a // g}/{b // g}" in tail:
                return True
        except ValueError:
            pass
    return False


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def run_code(src, fname, tests, timeout=15):
    prog = f"{src}\n\n{tests}\nprint('CAPOK')\n"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(prog)
        try:
            r = subprocess.run([sys.executable, p], capture_output=True, timeout=timeout,
                               text=True, cwd=d)
            return "CAPOK" in r.stdout, (r.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--sections", default="reasoning,code")
    args = ap.parse_args()
    sections = args.sections.split(",")
    out = {"tag": args.tag}

    if "reasoning" in sections:
        det, sc = [], 0
        for q, want in REASONING:
            try:
                a = ask(args.port, args.model, q)
                ok = check_reasoning(a, want)
            except Exception as e:  # noqa: BLE001
                a, ok = f"ERROR {e}", False
            sc += ok
            det.append({"q": q[:70], "want": want, "got": a.strip()[-120:], "ok": bool(ok)})
            print(("PASS" if ok else "FAIL"), "| reason |", q[:56], "->", want)
        out["reasoning"] = {"score": sc, "total": len(REASONING), "details": det}
        print(f"\nREASONING: {sc}/{len(REASONING)}")

    if "code" in sections:
        det, sc = [], 0
        for q, fname, tests in CODE:
            try:
                a = ask(args.port, args.model, q, max_tokens=1400)
                ok, err = run_code(extract_code(a), fname, tests)
            except Exception as e:  # noqa: BLE001
                ok, err = False, str(e)[:200]
            sc += ok
            det.append({"fn": fname, "ok": bool(ok), "err": "" if ok else err})
            print(("PASS" if ok else "FAIL"), "| code   |", fname, "" if ok else f"({err[:70]})")
        out["code"] = {"score": sc, "total": len(CODE), "details": det}
        print(f"\nCODE: {sc}/{len(CODE)}")

    tot = sum(out[s]["score"] for s in out if isinstance(out.get(s), dict) and "score" in out[s])
    mx = sum(out[s]["total"] for s in out if isinstance(out.get(s), dict) and "total" in out[s])
    out["combined"] = {"score": tot, "total": mx}
    print(f"\nCOMBINED: {tot}/{mx}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
