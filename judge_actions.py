# -*- coding: utf-8 -*-
"""
LLM-as-Judge for MT-Bench answers across multiple actions (models/quantizations).

Inputs:
  --questions  path to questionset.jsonl.txt   (e.g. C:\...\questionset.jsonl.txt)
  --runs_root  root dir containing per-action subdirs with latest run folders
               (e.g. E:\SLM\runs_mtbench2)
  --model      judge model name (default: gpt-4o)
  --api_url    chat completion endpoint (default: https://api.chatanywhere.tech/v1/chat/completions)
  --api_key_env  env var name for API key (default: CHATANY_API_KEY)
  --turn       which MT-Bench turn to judge (1 or 2, default 1)
  --out        output CSV (default: scores_mtbench.csv)

Directory layout expected under runs_root:
  runs_root/
    <ACTION_NAME>/
      <ACTION_NAME>__mtbench_YYYYmmdd_HHMMSS/
        answers.jsonl   # preferred if present
        q_0001_.../answer.txt
        q_0002_.../answer.txt
        ...

Output:
  CSV with rows = question_id, columns = action names, values = judge scores (0..10)
"""

import os, json, time, re, argparse
from pathlib import Path
import requests
import csv
from datetime import datetime

# ---------- Prompt template (as given) ----------
PROMPT_TMPL = (
    "You are an expert evaluator. Rate response quality on a scale of 0-10.\n\n"
    "<|user|>\n"
    "Evaluate how well this response addresses the request (0-10 scale):\n\n"
    'Original Request: "{original_prompt}"\n'
    "Category: {category}\n"
    'Response: "{final_response}..."\n\n'
    "Criteria:\n"
    "1. Relevance (0-3): Addresses the request?\n"
    "2. Completeness (0-3): Adequate information?\n"
    "3. Usefulness (0-2): Helpful to user?\n"
    "4. Coherence (0-2): Well-organized?\n\n"
    "Provide only a single number score (0-10). No explanation.\n\n"
    "<|assistant|>"
)

def load_questions(questions_path: Path, turn: int):
    """Return list of dicts: {question_id, category, prompt} for chosen turn (1/2)."""
    data = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            cat = obj.get("category") or "NA"
            turns = obj.get("turns") or []
            idx = max(0, min(len(turns), turn) - 1)  # turn is 1-based
            prompt = turns[idx] if turns else ""
            data.append({"question_id": qid, "category": cat, "prompt": prompt})
    return data

def find_latest_run_dir(action_dir: Path):
    """Pick the latest *_mtbench_YYYYmmdd_HHMMSS directory under an action dir."""
    if not action_dir.exists():
        return None
    candidates = [p for p in action_dir.iterdir() if p.is_dir() and "__mtbench_" in p.name]
    if not candidates:
        return None
    # sort by mtime, descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def load_answers_from_run(run_dir: Path):
    """
    Return mapping: {index(int or question_id): answer_text, 'qdir': Path}
    Prefer answers.jsonl if present; fallback to reading q_*/answer.txt by index order.
    """
    answers = {}
    ans_jsonl = run_dir / "answers.jsonl"
    if ans_jsonl.exists():
        with open(ans_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "error" in rec:
                    continue
                # prefer question_id if present; else fall back to index
                key = rec.get("question_id") or rec.get("index")
                # we have explicit paths:
                ptxt = rec.get("answer_path_txt")
                if ptxt and Path(ptxt).exists():
                    try:
                        answers[key] = Path(ptxt).read_text(encoding="utf-8")
                    except Exception:
                        pass
    else:
        # fallback: read q_* folders in name order
        q_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("q_")])
        for idx, qd in enumerate(q_dirs, start=1):
            ans = qd / "answer.txt"
            if ans.exists():
                try:
                    answers[idx] = ans.read_text(encoding="utf-8")
                except Exception:
                    pass
    return answers

def call_judge(api_url, api_key, model, prompt_text, timeout=60):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        # 强烈建议让回复更短，减少误差
        "temperature": 0.0,
        "max_tokens": 10,
    }
    resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # 兼容 openai 样式
    msg = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return msg.strip()

def parse_score(text):
    """
    Extract a 0..10 integer/float from LLM response (which should be a single number).
    Robust to stray characters.
    """
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
        # clamp
        if val < 0: val = 0.0
        if val > 10: val = 10.0
        return val
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, required=True, help="questionset.jsonl.txt")
    ap.add_argument("--runs_root", type=str, required=True, help="root dir with action subdirs")
    ap.add_argument("--model", type=str, default="gpt-4o", help="judge model name")
    ap.add_argument("--api_url", type=str, default="https://api.chatanywhere.tech/v1/chat/completions")
    ap.add_argument("--api_key_env", type=str, default="CHATANY_API_KEY")
    ap.add_argument("--turn", type=int, default=1, choices=[1,2], help="which turn of MT-Bench to judge (1 or 2)")
    ap.add_argument("--out", type=str, default="scores_mtbench.csv")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep between API calls (sec)")
    ap.add_argument("--retries", type=int, default=3, help="retries per call")
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in env var {args.api_key_env}")

    questions_path = Path(args.questions)
    runs_root = Path(args.runs_root)
    out_csv = Path(args.out)

    # 1) load questions
    qlist = load_questions(questions_path, args.turn)  # [{question_id, category, prompt}]
    # build dict by question_id and also by ordinal index
    by_qid = {q["question_id"]: q for q in qlist}
    # some pipelines may only have index keys; keep an ordered list of qids
    ordered_qids = [q["question_id"] for q in qlist]

    # 2) discover actions
    action_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    action_dirs.sort()
    actions = []
    for adir in action_dirs:
        latest = find_latest_run_dir(adir)
        if latest:
            actions.append((adir.name, latest))
    if not actions:
        raise SystemExit(f"No action runs found under: {runs_root}")

    print("Found actions:")
    for name, run in actions:
        print(f"  - {name}: {run}")

    # 3) pre-load answers for each action
    answers_per_action = {}  # action_name -> dict key(question_id or index) -> answer
    for name, run in actions:
        answers = load_answers_from_run(run)
        answers_per_action[name] = answers

    # 4) scoring
    # CSV header: question_id, then each action name
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        header = ["question_id"] + [name for name, _ in actions]
        writer.writerow(header)

        sess = requests.Session()
        for qid in ordered_qids:
            row = [qid]
            # pull question text/category
            qrec = by_qid.get(qid)
            if not qrec:
                # skip if not found
                # still write an empty row to keep alignment
                for _ in actions:
                    row.append("")
                writer.writerow(row)
                continue

            original_prompt = (qrec["prompt"] or "").strip()
            category = qrec["category"] or "NA"

            for name, _run in actions:
                # answers may be keyed by question_id or by index (1-based)
                answers = answers_per_action.get(name, {})
                ans_text = None
                # try by qid
                if qid in answers:
                    ans_text = answers[qid]
                else:
                    # fallback: by ordinal index (position in ordered_qids)
                    idx1 = ordered_qids.index(qid) + 1  # 1-based
                    ans_text = answers.get(idx1)

                if not ans_text:
                    row.append("")
                    continue

                final_resp = (ans_text.replace("\n", " ").strip())[:400]
                prompt = PROMPT_TMPL.format(
                    original_prompt=original_prompt,
                    category=category,
                    final_response=final_resp
                )

                score = None
                for t in range(args.retries):
                    try:
                        reply = call_judge(args.api_url, api_key, args.model, prompt)
                        score = parse_score(reply)
                        if score is not None:
                            break
                    except Exception as e:
                        if t == args.retries - 1:
                            print(f"[{name} qid={qid}] FAILED after {args.retries} tries: {e}")
                    time.sleep(args.sleep)

                row.append("" if score is None else f"{score:.2f}")

            writer.writerow(row)

    print(f"\nDone. Wrote: {out_csv.resolve()}")
    print("Tip: open in Excel/Sheets to view the action-wise score matrix.")

if __name__ == "__main__":
    main()
