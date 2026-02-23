#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, request, Response, send_file, abort

# -----------------------------
# Parsing helpers
# -----------------------------



def normalize_letter(x: Any) -> Optional[str]:
    """
    Normalize experiment predictions and correct_option to uppercase letters A-H (or None).
    Accepts inputs like: "A", "A.", " a ", "B)", etc.
    """
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    m = re.match(r"^([A-H])\s*[\.\)]?\s*$", s)
    if m:
        return m.group(1)
    # If input like "A. foo", take first letter if it looks like an option prefix
    m2 = re.match(r"^([A-H])\s*[\.\)]\s+.+", s)
    if m2:
        return m2.group(1)
    return None


def parse_options(raw: Any) -> Tuple[List[str], str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ([], "")
    s = str(raw).strip()
    if not s:
        return ([], "")

    # Try JSON list (handles doubled quotes typical of CSV)
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return ([str(x) for x in j], s)
    except Exception:
        pass

    # Try python literal list
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return ([str(x) for x in v], s)
    except Exception:
        pass

    # Delimiters: prefer || then | then ; then newline
    for delim in ["\n", "||", "|", ";"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return (parts, s)

    # "A. foo B. bar ..." pattern
    letter_splits = re.split(r"(?=(?:^|\s)[A-Ha-h]\.\s)", s)
    letter_splits = [p.strip() for p in letter_splits if p.strip()]
    if len(letter_splits) >= 2:
        return (letter_splits, s)

    return ([s], s)


def letter_to_index(letter: Optional[str]) -> Optional[int]:
    if not letter:
        return None
    return ord(letter) - ord("A")


def option_text_for_letter(options_list: List[str], letter: Optional[str]) -> Optional[str]:
    idx = letter_to_index(letter)
    if idx is None:
        return None
    if 0 <= idx < len(options_list):
        # Strip "A. " prefixes for readability
        t = str(options_list[idx])
        t = re.sub(r"^\s*[A-Ha-h]\s*[\.\)]\s*", "", t).strip()
        return t
    return None


def guess_exp_cols(columns: List[str]) -> List[str]:
    # Default: anything that starts with "exp" and is not the ground-truth correct_option
    exp_cols = []
    for c in columns:
        cl = c.lower()
        if cl == "correct_option":
            continue
        if cl.startswith("exp"):
            exp_cols.append(c)

    def key_fn(name: str) -> Tuple[int, str]:
        m = re.search(r"(\d+)", name)
        return (int(m.group(1)) if m else 10**9, name)

    exp_cols.sort(key=key_fn)
    return exp_cols


# -----------------------------
# Server
# -----------------------------

def load_flags(flags_path: str) -> Dict[str, bool]:
    if os.path.exists(flags_path):
        try:
            with open(flags_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # keys -> true
                return {k: bool(v) for k, v in data.items() if v}
        except Exception:
            pass
    return {}


def save_flags(flags_path: str, flags: Dict[str, bool]) -> None:
    tmp = flags_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=2)
    os.replace(tmp, flags_path)


def build_exp_filters_html(exp_cols: List[str]) -> str:
    blocks = []
    for c in exp_cols:
        blocks.append(
            f"""
            <div class="control">
              <label>{c} correctness</label>
              <select id="filter__{c}">
                <option value="all" selected>All</option>
                <option value="correct">Correct</option>
                <option value="incorrect">Incorrect</option>
              </select>
            </div>
            """.strip()
        )
    return "\n".join(blocks)


def build_sort_exp_options(exp_cols: List[str]) -> str:
    return "\n".join([f'<option value="{c}">{c} (correctness)</option>' for c in exp_cols])


def make_app(csv_path: str, template_path: str, title: str, exp_cols_cli: Optional[List[str]], flags_path: str) -> Flask:
    app = Flask(__name__)
    flags = load_flags(flags_path)

    # Load CSV once at startup (fast enough for typical sizes).
    df = pd.read_csv(csv_path)

    required = ["dataset", "question_id", "video_path", "question", "options", "correct_option", "task_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    exp_cols = exp_cols_cli if exp_cols_cli else guess_exp_cols(list(df.columns))
    if not exp_cols:
        raise SystemExit("No experiment prediction columns found. Provide them via --exp_cols col1 col2 ...")

    # Preprocess into row objects for frontend
    rows: List[Dict[str, Any]] = []
    for i, r in df.iterrows():
        options_list, options_raw = parse_options(r.get("options"))
        correct_letter = normalize_letter(r.get("correct_option"))
        correct_idx = letter_to_index(correct_letter)

        exp_pred: Dict[str, Optional[str]] = {}
        exp_correct: Dict[str, Optional[bool]] = {}
        exp_choice_text: Dict[str, Optional[str]] = {}

        for c in exp_cols:
            pred_letter = normalize_letter(r.get(c))
            exp_pred[c] = pred_letter
            exp_correct[c] = (pred_letter == correct_letter) if (pred_letter and correct_letter) else None
            exp_choice_text[c] = option_text_for_letter(options_list, pred_letter)

        rows.append({
            "__index": int(i),
            "dataset": "" if pd.isna(r.get("dataset")) else str(r.get("dataset")),
            "task_type": "" if pd.isna(r.get("task_type")) else str(r.get("task_type")),
            "question_id": "" if pd.isna(r.get("question_id")) else str(r.get("question_id")),
            "video_path": "" if pd.isna(r.get("video_path")) else str(r.get("video_path")),
            "question": "" if pd.isna(r.get("question")) else str(r.get("question")),
            "options_raw": "" if pd.isna(options_raw) else str(options_raw),
            "options_list": options_list,
            "correct_option": correct_letter if correct_letter else ("" if pd.isna(r.get("correct_option")) else str(r.get("correct_option"))),
            "correct_index": correct_idx,
            "exp_pred": exp_pred,
            "exp_correct": exp_correct,
            "exp_choice_text": exp_choice_text,
        })

    # HTML template injection
    with open(template_path, "r", encoding="utf-8") as f:
        tpl = f.read()

    exp_filters_html = build_exp_filters_html(exp_cols)
    sort_exp_options = build_sort_exp_options(exp_cols)

    html_page = (
        tpl.replace("{{TITLE}}", title)
           .replace("{{EXP_FILTERS}}", exp_filters_html)
           .replace("{{SORT_EXP_OPTIONS}}", sort_exp_options)
           .replace("{{EXP_COLS_JSON}}", json.dumps(exp_cols, ensure_ascii=False))
    )

    @app.get("/")
    def index():
        return Response(html_page, mimetype="text/html")

    @app.get("/api/data")
    def api_data():
        return jsonify(rows)

    @app.get("/api/flags")
    def api_flags():
        return jsonify({"flags": flags})

    @app.post("/api/flag")
    def api_flag():
        nonlocal flags
        payload = request.get_json(force=True, silent=True) or {}
        key = payload.get("key")
        flagged = payload.get("flagged")

        if not isinstance(key, str) or not key:
            return jsonify({"error": "Invalid key"}), 400

        if bool(flagged):
            flags[key] = True
        else:
            flags.pop(key, None)

        save_flags(flags_path, flags)
        return jsonify({"flags": flags})
    @app.get("/api/video")
    def api_video():
        path = request.args.get("path", "")
        if not path:
            abort(400, "Missing path")

        # Absolute path on the server filesystem
        real_path = os.path.realpath(path)

        # SECURITY: you must restrict what can be served.
        # Set VIDEO_ALLOW_ROOTS to a colon-separated list of allowed directories:
        # e.g. /scratch/datasets:/home/user/videos
        allow_roots = os.environ.get("VIDEO_ALLOW_ROOTS", "")
        if not allow_roots:
            abort(500, "VIDEO_ALLOW_ROOTS not set")

        allowed = False
        for root in allow_roots.split(":"):
            root = root.strip()
            if not root:
                continue
            real_root = os.path.realpath(root)
            if real_path == real_root or real_path.startswith(real_root + os.sep):
                allowed = True
                break

        if not allowed:
            abort(403, "Path not allowed")

        if not os.path.exists(real_path):
            abort(404, "Video not found")

        # conditional=True enables range/conditional requests in many cases (helps seeking).
        return send_file(real_path, mimetype="video/mp4", conditional=True)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--template", default="template_2.html", help="Path to template.html")
    ap.add_argument("--title", default="Video QA Report", help="Report title")
    ap.add_argument("--exp_cols", nargs="*", default=None,
                    help="Explicit experiment prediction columns (space-separated). Example: --exp_cols exp1_pred exp2_pred exp3_pred")
    ap.add_argument("--flags", default="flags.json", help="Where to persist flags (JSON)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = make_app(args.csv, args.template, args.title, args.exp_cols, args.flags)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
