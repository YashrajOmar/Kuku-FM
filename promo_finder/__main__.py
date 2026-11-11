"""Run:  python -m promo_finder --data data --provider gemini --out outputs/run1"""
import argparse

from .llm import LLMClient, MockClient
from .pipeline import run, write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Find 20-60s promo clips in microdrama screenplays.")
    ap.add_argument("--data", default="data", help="folder of episode .txt screenplays")
    ap.add_argument("--provider", default="gemini", help="gemini | groq | openrouter | openai | custom | mock")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--out", default="outputs/run1")
    ap.add_argument("--rubric", default="rubric.yaml")
    ap.add_argument("--episodes", default=None, help="e.g. 1-5,12 (default: all)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--list-models", action="store_true", help="print the model names your provider and key can use, then exit")
    ap.add_argument("--stitch", action="store_true", help="also try promos that join moments from different episodes (1 extra AI call)")
    a = ap.parse_args()

    if a.list_models:
        print("\n".join(LLMClient(a.provider, a.model, cache_dir=None).list_models()))
        return
    client = MockClient() if a.provider == "mock" else LLMClient(a.provider, a.model, cache_dir=None if a.no_cache else ".cache")
    ranked, meta = run(a.data, client, a.rubric, a.episodes, stitch=a.stitch)
    write_outputs(ranked, meta, a.out)
    print(f"\n{meta['promos_after_dedupe']} promos from {meta['episodes_processed']} episodes -> {a.out}/ "
          f"({meta['tokens_in']} tokens in, {meta['tokens_out']} out, {meta['seconds']}s)")
    for p in ranked[:5]:
        print(f"  #{p['rank']} {p['total_score']:>5}  Ep{p['episode']}  {p['segments'][0]['start_ts']}-{p['segments'][-1]['end_ts']}  {p['hook_type']}")


if __name__ == "__main__":
    main()