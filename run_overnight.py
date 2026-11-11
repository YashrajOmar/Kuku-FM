"""Run ALL episodes unattended. Safe to stop (Ctrl+C) and start again: finished episodes are cached, so they cost nothing.

    python run_overnight.py --provider groq --model openai/gpt-oss-120b --out outputs/run_full

What it does
  1. Processes every episode. The results so far are written to <out>/ after EVERY round, so you always have something.
  2. If the provider's daily limit stops it, it waits and tries again (episodes already done come from the cache).
  3. When every episode is done, it runs the cross-episode pass into <out>_stitch/. That pass is kept in its own
     folder so your main pack stays clean. Every join is checked by a separate "cold viewer" AI call.
  4. Everything it does is logged to <out>/overnight_log.txt.

Quick plumbing test with no API and no cost:  python run_overnight.py --provider mock --out outputs/mock_test
Windows: it asks Windows not to go to sleep while it runs. Closing the laptop lid can still put it to sleep.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from promo_finder.llm import LLMClient, MockClient
from promo_finder.pipeline import run, write_outputs


def keep_awake() -> None:
    """Windows only: tell the OS the computer is busy so it does not sleep (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def say(msg: str, log: Path) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    try:   # model replies can contain characters the Windows console cannot print
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Unattended full run with automatic waiting on daily limits.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--provider", default="groq", help="groq | gemini | openrouter | openai | custom | mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rubric", default="rubric.yaml")
    ap.add_argument("--out", default="outputs/run_full")
    ap.add_argument("--episodes", default=None, help="e.g. 1-20 (default: all)")
    ap.add_argument("--wait-minutes", type=float, default=30, help="pause between rounds when the daily limit stops a round")
    ap.add_argument("--max-hours", type=float, default=30, help="give up waiting after this many hours")
    ap.add_argument("--no-stitch", action="store_true", help="skip the cross-episode pass")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "overnight_log.txt"
    keep_awake()
    client = MockClient() if a.provider == "mock" else LLMClient(a.provider, a.model, cache_dir=".cache")
    say(f"Start. model = {client.label}", log)

    started, rounds, error_rounds, done = time.time(), 0, 0, False
    try:
        while True:
            rounds += 1
            say(f"Round {rounds}: processing episodes (finished ones come from the cache)", log)
            ranked, meta = run(a.data, client, a.rubric, a.episodes)
            write_outputs(ranked, meta, str(out))          # always leave the latest results on disk
            left, errs = meta["episodes_not_done"], meta["errors"]
            say(f"Round {rounds} result: {meta['promos_after_dedupe']} promos | {len(left)} episode(s) not reached | "
                f"{len(errs)} error(s) | tokens in/out (cached ones included): {meta['tokens_in']}/{meta['tokens_out']}", log)
            if not left and not errs:
                done = True
                break
            if not left:                                   # only real errors left, not the daily limit
                error_rounds += 1
                if error_rounds >= 3:
                    say("Giving up on the failing episodes. See 'errors' in run_meta.json for the reason.", log)
                    break
            if time.time() - started + a.wait_minutes * 60 > a.max_hours * 3600:
                say("Time limit reached. Run the same command again later to continue; finished episodes are cached.", log)
                break
            say(f"Waiting {a.wait_minutes:g} minutes, then trying again. (You can press Ctrl+C any time; results so far are saved.)", log)
            time.sleep(a.wait_minutes * 60)

        if done and not a.no_stitch:
            say("All episodes done. Starting the cross-episode pass (each join is checked by a cold-viewer judge).", log)
            sdir = Path(str(out) + "_stitch")
            ranked, meta = run(a.data, client, a.rubric, a.episodes, stitch=True)
            write_outputs(ranked, meta, str(sdir))
            st = meta["stitch"]
            joined = sum(1 for p in ranked if p.get("cross_episode"))
            say(f"Cross-episode pass: kept {joined} joined promo(s) in {sdir}. Dropped joins: {st.get('dropped_joins', [])}. "
                f"Note/error: {st.get('note') or st.get('error') or '-'}", log)
    except KeyboardInterrupt:
        say("Stopped by you (Ctrl+C). Results from the last finished round are in " + str(out), log)
        return

    say("DONE. Send me: promos.md, run_meta.json and overnight_log.txt from " + str(out), log)


if __name__ == "__main__":
    main()