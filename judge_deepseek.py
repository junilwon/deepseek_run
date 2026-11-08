"""EgoTempo OpenQA judging with DeepSeek-V3 (OpenAI-compatible API), standard library only (no pip install needed).

Judge prompt = the official EgoTempo judge prompt verbatim (google-research-datasets/egotempo, gemini_eval.ipynb cell 13),
sent as one user message, temperature 0. A prediction counts as correct when the judge's dict has pred == 'correct'.
Unparsable judge output counts as incorrect and is reported. Resumable: finished rows in results/<model>.jsonl are skipped,
rows with API errors are retried on the next run.

usage:
  (key: DEEPSEEK_API_KEY env var, else deepseek_api_key.txt next to this script)
  python judge_deepseek.py                 # all models in data/
  python judge_deepseek.py --models qwen3vl8b_base --workers 8
  python judge_deepseek.py --dry_run       # print the first prompt, no API call
outputs: results/<model>.jsonl (raw judge text + parsed dict), results/summary.json, results/summary.md
"""
import argparse, ast, collections, glob, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None, help="data/<name>.jsonl names (default: all)")
ap.add_argument("--judge_model", default="deepseek-chat", help="DeepSeek model id (deepseek-chat = DeepSeek-V3)")
ap.add_argument("--base_url", default="https://api.deepseek.com")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=256)
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))


def create_prompt(q, a, pred):   # verbatim from the official notebook
    return f"""role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {q}
Ground truth correct Answer: {a}
Predicted Answer: {pred}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""


def parse(text):
    m = re.search(r"\{.*?\}", text or "", re.DOTALL)
    if not m: return None
    try:
        d = ast.literal_eval(m.group(0))
        return {"pred": str(d.get("pred", "")).strip().lower(), "score": int(d.get("score", 0)), "reason": str(d.get("reason", ""))}
    except Exception:
        return None



def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out


def call(prompt, key):
    body = json.dumps({"model": args.judge_model, "temperature": 0, "max_tokens": args.max_tokens, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"], d.get("model")


def judge_row(r, key):
    raw, used, err = None, None, None
    for attempt in range(4):
        try:
            raw, used = call(create_prompt(r["question"], r["answer"], r["prediction"]), key); err = None; break
        except Exception as e:   # rate limit / network: back off and retry
            err = f"{type(e).__name__}: {str(e)[:200]}"; time.sleep(5 * (attempt + 1))
    return {"question_id": r["question_id"], "question_type": r["question_type"], "judge_model": used or args.judge_model,
            "judge_raw": raw, "parsed": parse(raw) if raw else None, "judge_error": err}


models = args.models or sorted(os.path.basename(f)[:-6] for f in glob.glob(os.path.join(HERE, "data", "*.jsonl")))
if args.dry_run:
    r = read_jsonl(os.path.join(HERE, "data", models[0] + ".jsonl"))[0]
    print(create_prompt(r["question"], r["answer"], r["prediction"])); sys.exit(0)
kf = os.path.join(HERE, "deepseek_api_key.txt")   # private repo: key file kept here at the user's request
key = os.environ.get("DEEPSEEK_API_KEY") or (open(kf).read().strip() if os.path.exists(kf) else None) or sys.exit("set DEEPSEEK_API_KEY or add deepseek_api_key.txt")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
summary = {}
for m in models:
    rows = {r["question_id"]: r for r in read_jsonl(os.path.join(HERE, "data", m + ".jsonl"))}
    rf = os.path.join(HERE, "results", m + ".jsonl")
    done = {r["question_id"]: r for r in read_jsonl(rf) if r.get("judge_error") is None}
    todo = [r for q, r in rows.items() if q not in done]
    print(f"[{m}] {len(rows)} predictions, {len(done)} judged, {len(todo)} to do", flush=True)
    with open(rf, "a", encoding="utf-8") as fo, ThreadPoolExecutor(args.workers) as ex:
        for k, rec in enumerate(ex.map(lambda r: judge_row(r, key), todo)):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            if rec["judge_error"] is None: done[rec["question_id"]] = rec
            if (k + 1) % 50 == 0: print(f"  {k+1}/{len(todo)}", flush=True)
    by = collections.defaultdict(lambda: [0, 0]); unparsable = 0
    for q, rec in done.items():
        if q not in rows: continue
        ok = bool(rec["parsed"]) and rec["parsed"]["pred"] == "correct"; unparsable += rec["parsed"] is None
        by[rec["question_type"]][0] += ok; by[rec["question_type"]][1] += 1
    n = sum(v[1] for v in by.values())
    summary[m] = {"n_judged": n, "n_predictions": len(rows), "errors_pending": len(rows) - n, "unparsable": unparsable,
                  "overall": round(100 * sum(v[0] for v in by.values()) / max(1, n), 1),
                  "per_category": {c: f"{a}/{t} ({100*a/t:.1f})" for c, (a, t) in sorted(by.items())},
                  "judge_model": sorted({r.get("judge_model") for r in done.values()})}
    print(f"  {m}: overall {summary[m]['overall']} on {n}/{len(rows)} (unparsable {unparsable})", flush=True)
json.dump(summary, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=2, ensure_ascii=False)
cats = sorted({c for s in summary.values() for c in s["per_category"]})
with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
    f.write("| model | overall | n | " + " | ".join(cats) + " |\n|" + "---|" * (len(cats) + 3) + "\n")
    for m, s in summary.items():
        f.write(f"| {m} | {s['overall']} | {s['n_judged']} | " + " | ".join(s["per_category"].get(c, "-").split(" ")[0] for c in cats) + " |\n")
print(open(os.path.join(HERE, "results", "summary.md")).read())
