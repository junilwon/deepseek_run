# EgoTempo answers + DeepSeek-V3 judge

Model answers on the EgoTempo OpenQA benchmark (130 of the 500 questions: the ones whose Ego4D clips we have), packaged so the
LLM-judge step can run on any machine that can reach `api.deepseek.com`.

## Contents
- `data/<model>.jsonl` — one line per question: `question_id`, `question_type`, `question`, `answer` (ground truth), `prediction`.
  Models: Qwen3-VL-8B base, Qwen3-VL-8B + mix v3 SFT (ck2099), lf-c0-attn stage 1 + mix v3 SFT (ck1500), lf-c0-attn stage 1 only,
  InternVL3.5-8B base. Inference: 32 uniformly sampled frames of each clip, official EgoTempo answer prompt, greedy decoding
  (`data/MODELS.json` lists the source run of each file).
- `judge_deepseek.py` — judge with the official EgoTempo judge prompt (verbatim from the egotempo repo `gemini_eval.ipynb`),
  DeepSeek-V3 (`deepseek-chat`), temperature 0. Python 3 standard library only.

## Run
```bash
python judge_deepseek.py            # key is read from deepseek_api_key.txt (or DEEPSEEK_API_KEY)            # all models; resumable (re-run to retry errors)
```
Cheap trial first: `python judge_deepseek.py --limit 5` (5 new rows per model), then run without `--limit`.
Interruption-safe: every finished call is appended and flushed to `results/<model>.jsonl` at once; Ctrl-C cancels the calls not yet
sent, so at most `--workers` (default 8) in-flight calls are lost. Re-running skips everything already saved and retries API errors.
Token counts are printed every 25 calls and at the end.
Results go to `results/<model>.jsonl` (raw judge text + parsed dict) and `results/summary.json` / `results/summary.md`
(overall accuracy and per-category counts). Commit and push `results/` back when done.

## Notes
- The paper's judge is Gemini 1.5 Pro; this repo uses DeepSeek-V3 for all models, so the numbers compare only with each other.
- The 130 questions are not balanced across the 10 categories, so `overall` here is not the paper's category average.
