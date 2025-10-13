#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

def extract_function_name(code: str) -> str:
    m = re.search(r'\b([A-Za-z_]\w*)\s*\([^)]*\)\s*{?', code)
    return m.group(1) if m else "unknown"

def sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w.-]", "_", s)
    return s[:200] or "row"

def parse_annotations(text: str):
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = text.splitlines()
    i = 0
    ann = {}
    if i < len(lines) and lines[i].strip().lower() in {"c", "c++"}:
        i += 1
    while i < len(lines):
        if lines[i].strip().upper().startswith("LINE "):
            m = re.match(r"\s*LINE\s+(\d+)\s*$", lines[i], re.IGNORECASE)
            i += 1
            if not m:
                continue
            ln = int(m.group(1))
            snippet_lines = []
            while i < len(lines) and not re.match(r"\s*LINE\s+\d+\s*$", lines[i], re.IGNORECASE):
                snippet_lines.append(lines[i])
                i += 1
            while snippet_lines and snippet_lines[0].strip() == "":
                snippet_lines.pop(0)
            snippet = ("\n".join(snippet_lines)).rstrip("\n")
            if snippet:
                ann.setdefault(ln, []).append(snippet)
        else:
            i += 1
    return ann

def strip_line_numbers(code: str) -> str:
    return re.sub(r'^\s*\d+\s+', '', code, flags=re.MULTILINE)

def apply_annotations(code: str, annotations_text: str) -> str:
    ann = parse_annotations(annotations_text)
    if not ann:
        return strip_line_numbers(code)
    src = code.splitlines()
    out = []
    for idx, line in enumerate(src, start=1):
        out.append(line)
        if (idx - 1) in ann:
            for snippet in reversed(ann[idx - 1]):
                out.extend(snippet.splitlines())
    return strip_line_numbers("\n".join(out) + ("\n" if code.endswith("\n") else ""))

def main():
    ap = argparse.ArgumentParser(description="Export C files from parquet rows.")
    ap.add_argument("parquet", help="Path to parquet file (e.g., export_25-06-05.parquet)")
    ap.add_argument("-n", "--max-rows", type=int, default=0,
                    help="Number of rows to process; 0 means all (default: 0)")
    ap.add_argument("--positive-scores", nargs="+", required=True,
                    help="Scores categorized as label 1 (no default; must be provided)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    num_rows = len(df)
    scores_tag = "_".join(args.positive_scores)
    outdir = Path("raw_code") / f"{scores_tag}_{num_rows}"
    outdir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for _, row in df.iterrows():
        score = str(row["score"])
        label = 1 if score in args.positive_scores else 0
        code = str(row["code"])
        if "annotation" in df.columns and isinstance(row["annotation"], str):
            code = apply_annotations(code, row["annotation"])
        else:
            code = strip_line_numbers(code)
        base = sanitize_name(extract_function_name(code))
        counts[base] = counts.get(base, 0) + 1
        (outdir / f"{base}_{counts[base]}_{label}.c").write_text(code, encoding="utf-8")

if __name__ == "__main__":
    main()
