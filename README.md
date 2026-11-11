# Kuku Promo Finder

Reads microdrama screenplays and finds the best 20-60s promo clips (hook in the first 3s, cliffhanger ending, understandable on its own), tagged with 12 human-tension categories and ranked across all episodes.

The AI judges taste (hook, cliffhanger, standalone clarity, scene meaning and themes, by reading the scene, never by keyword matching). Code guarantees correctness: real timestamps from line ids, the 20-60s rule, no overlaps, quotes copied from the script, tags kept only if they cite a real line in the clip, and scores computed from `rubric.yaml`.

Result on the 50-episode show: 50 promos; on 15 episodes I labelled by hand, the tool's pick overlapped mine in 14/15 (median overlap 0.90). See `eval/report.md`.

## Setup
```
pip install -r requirements.txt
copy .env.example .env      (Windows; on Mac/Linux: cp .env.example .env)
```
Put your key in `.env`, e.g. `GROQ_API_KEY=...`. Never commit it.
Put the episode `.txt` files in `./data` (the input pack is confidential, so `data/` and `outputs/` are git-ignored).

## Run
```
python -m pytest -q                                                              # tests, no API needed
python -m promo_finder --provider mock --out outputs/mock                        # offline plumbing check
python -m promo_finder --provider groq --model openai/gpt-oss-120b --episodes 1-5 --out outputs/try   # small real run first
python run_overnight.py --provider groq --model openai/gpt-oss-120b --out outputs/run_full            # all episodes; waits out daily limits and resumes
streamlit run app.py                                                             # web app with the 12-category filter
python evaluate.py template                                                      # hand-label template (then fill it in)
python evaluate.py score outputs/run_full/promos.json eval/labels.csv            # hit rate + failure list
```
Outputs: `promos.md` (readable), `promos.csv`, `promos.json` (full detail + transcript), `run_meta.json` (model, tokens, rejected AI proposals, errors).

## Swap the model (no code change)
`--provider groq | gemini | openrouter | openai | custom` and `--model <name>`. Any OpenAI-compatible endpoint works (`--provider custom` with `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). `--list-models` prints the model names your key can use. Replies are cached in `.cache/`, so a re-run is free and identical.

## Change the taste, not the code
`rubric.yaml`: weights, hook and cliffhanger types, the theme table (official labels, what each covers, reward, platform risk), the 12 categories, stitching limits.

## Known limits
- Works from screenplay text only, so it cannot see or hear the scene.
- Themes and scores come from the AI. Scores are bunched (most hooks get 8), so treat the top ~10 as a shortlist.
- One real miss: episode 46 (the tool picked the argument before my labelled ending beat).
- Cross-episode joins (`--stitch`) exist but are unvalidated: in the real run the checks rejected the one proposed join, so none are in the submitted pack.
- Groq's free tier allows about 200K tokens per day, so a full 50-episode run takes a few hours unattended.
