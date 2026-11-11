# Kuku Promo Finder

Reads microdrama screenplays and finds the best 20-60s promo clips (hook in the first 3s, cliffhanger ending,
understandable on its own), tagged with 12 human-tension categories and ranked across all episodes.

**AI judges taste** (hook, cliffhanger, standalone clarity, scene meaning and themes, by reading the scene, never by
keyword matching). **Code guarantees correctness**: real timestamps from line ids, 20-60s rule, no overlaps, quotes
copied from the script, tags kept only if they cite a real line in the clip, scores computed from `rubric.yaml`.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env (e.g. GEMINI_API_KEY=...). Never commit it.
# put the episode .txt files in ./data   (confidential: data/ and outputs/ are git-ignored)
```

## Run
```bash
python -m pytest -q                                                    # tests (no API needed)
python -m promo_finder --provider mock --out outputs/mock              # offline plumbing check
python -m promo_finder --provider gemini --episodes 1-5 --out outputs/try      # small real run first
python -m promo_finder --provider gemini --stitch --out outputs/run1           # full pack + cross-episode joins
streamlit run app.py                                                   # web app with the 12-category filter
python evaluate.py template                                            # hand-label template (then fill it in)
python evaluate.py score outputs/run1/promos.json eval/labels.csv      # hit rate + failure list
```
Outputs: `promos.md` (readable), `promos.csv`, `promos.json` (full detail + transcript), `run_meta.json`
(model, tokens, rejected AI proposals, episodes with no promo, errors).

## Swap the model (no code change)
`--provider gemini | groq | openrouter | openai | custom` and `--model <name>`. Any OpenAI-compatible endpoint works
(`--provider custom` with `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). Replies are cached in `.cache/`, so a re-run is free
and identical.

## Change the taste, not the code
`rubric.yaml`: weights, hook and cliffhanger types, the theme table (official labels, what each covers, reward,
platform risk), the 12 categories, stitching limits.

## Known limits
- Works from screenplay text only, so it cannot judge how a shot looks on screen.
- Themes and scores come from the AI; see `eval/report.md` for measured agreement with human picks.
- Cross-episode joins are limited to a small pool of top moments to keep cost low.
