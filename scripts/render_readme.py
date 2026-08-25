#!/usr/bin/env python3
"""Substitute generated tables into README.md.

The speed and quality tables are produced from results/*.json by summarize.py, so the
README can never drift from the data it claims to report. Run after any new experiment:

    python3 scripts/render_readme.py --ref Q-bf16
"""
import argparse, io, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generated(ref):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "summarize.py"), "--ref", ref],
        capture_output=True, text=True, check=True).stdout
    # summarize.py prints: speed table, blank, quality table, blank, ladder lines
    blocks, cur = [], []
    for line in out.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur)); cur = []
    if cur:
        blocks.append("\n".join(cur))
    if len(blocks) < 2:
        raise SystemExit("summarize.py did not produce two tables")
    return blocks[0], blocks[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Q-bf16")
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    a = ap.parse_args()
    speed, quality = generated(a.ref)
    src = io.open(a.readme, encoding="utf-8").read()
    for marker, table in (("<!--SPEED_TABLE-->", speed), ("<!--QUALITY_TABLE-->", quality)):
        if marker in src:
            src = src.replace(marker, table)
            continue
        # Already rendered once: swap the existing block, located by its header row and
        # extending over the consecutive table lines, so re-running after a new experiment
        # refreshes the numbers rather than failing or duplicating them.
        header = table.splitlines()[0]
        if header not in src:
            raise SystemExit(
                f"neither {marker} nor its header row found in {a.readme}; "
                "the table was edited by hand and cannot be refreshed safely"
            )
        lines = src.splitlines(keepends=True)
        i = next(n for n, ln in enumerate(lines) if ln.startswith(header))
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        src = "".join(lines[:i]) + table + "\n" + "".join(lines[j:])
    io.open(a.readme, "w", encoding="utf-8").write(src)
    print(f"rendered speed + quality tables into {a.readme} (ref={a.ref})")


if __name__ == "__main__":
    main()
