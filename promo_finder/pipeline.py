from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import yaml

from .extract import build_system_prompt, find_promos
from .llm import QuotaExhausted
from .parse import load_episodes


def load_rubric(path: str = "rubric.yaml") -> dict:
    rc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    w = sum(rc["weights"].values())
    if abs(w - 1.0) > 1e-6:
        raise ValueError(f"rubric weights must sum to 1.0, got {w}")
    return rc


def _overlap(a: dict, b: dict) -> bool:
    """True if two promos share more than half of the shorter one's time in the same episode."""
    for s in a["segments"]:
        for t in b["segments"]:
            if s["episode"] != t["episode"]:
                continue
            inter = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if inter > 0.5 * min(s["end"] - s["start"], t["end"] - t["start"]):
                return True
    return False


def rank(promos: list[dict]) -> list[dict]:
    """Best first. Ties go to the stronger cliffhanger. Near-duplicates of a better promo are dropped."""
    promos = sorted(promos, key=lambda p: (-p["total_score"], -p["scores"]["cliffhanger"]))
    kept: list[dict] = []
    for p in promos:
        if not any(_overlap(p, k) for k in kept):
            kept.append(p)
    for i, p in enumerate(kept, 1):
        p["rank"] = i
    return kept


def parse_range(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def run(data_dir: str, client, rubric_path: str = "rubric.yaml", episodes: str | None = None, verbose: bool = True,
        stitch: bool = False, progress=None):
    rc = load_rubric(rubric_path)
    system = build_system_prompt(rc)
    want = parse_range(episodes)
    eps = [e for e in load_episodes(data_dir) if not want or e.number in want]
    t0, promos, infos = time.time(), [], []
    quota_msg, not_done = None, []
    for n, ep in enumerate(eps, 1):
        if progress:
            progress(n, len(eps), ep.number)
        try:
            found, info = find_promos(ep, client, rc, system)
        except QuotaExhausted as e:  # daily allowance used up: stop cleanly, finished episodes stay cached for the next run
            quota_msg, not_done = str(e)[:300], [x.number for x in eps[n - 1:]]
            if verbose:
                print(f"Episode {ep.number:>2}: STOPPED, daily limit reached. {len(not_done)} episode(s) not processed. Run again later: finished ones are cached.")
            break
        except Exception as e:  # one bad episode must not kill the run; record it honestly
            found, info = [], {"episode": ep.number, "error": str(e)[:300]}
        promos += found
        infos.append(info)
        if verbose:
            why = ""
            if "error" in info:
                why = f"  ERROR {info['error']}"
            elif not found:   # say WHY an episode gave nothing, so you can see it without opening run_meta.json
                why = ("  | rejected: " + "; ".join(info.get("rejected", [])[:2])[:250]) if info.get("rejected") else "  | model proposed nothing"
            print(f"Episode {ep.number:>2}: {len(found)} promo(s){why}")
    ranked = rank(promos)
    sinfo: dict = {}
    if stitch and not quota_msg and len({e.number for e in eps}) > 1:   # optional 2nd pass: join moments from different episodes
        try:
            from .stitch import stitch_across_episodes
            extra, sinfo = stitch_across_episodes(ranked, eps, client, rc, system)
            ranked = rank(promos + extra)
        except Exception as e:
            sinfo = {"error": str(e)[:300]}
    meta = {
        "model": client.label,
        "episodes_processed": len(eps),
        "promos_found": len(promos),
        "promos_after_dedupe": len(ranked),
        "stitch": sinfo,
        "stopped_early": quota_msg,            # None when the run finished; otherwise why it stopped
        "episodes_not_done": not_done,         # episodes the run never reached (daily limit)
        "episodes_with_no_promo": [i["episode"] for i in infos if not any(p["episode"] == i["episode"] for p in ranked)],
        "errors": [i for i in infos if "error" in i],
        "rejected_proposals": {i["episode"]: i["rejected"] for i in infos if i.get("rejected")},
        "tokens_in": sum(i.get("tokens_in", 0) for i in infos),
        "tokens_out": sum(i.get("tokens_out", 0) for i in infos),
        "cached_calls": sum(i.get("cached", 0) for i in infos),
        "seconds": round(time.time() - t0, 1),
        "system_prompt_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
        "rubric": rc,
    }
    return ranked, meta


def _seg_str(p: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def write_outputs(ranked: list[dict], meta: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["rank", "episode", "segments", "duration_sec", "total_score", "emotion", "scene_summary", "categories", "themes", "hook_type", "hook", "hook_script_first_3s",
            "cliffhanger_type", "cliffhanger", "cliffhanger_script_last_lines", "why_it_works", "platform_risk", "risk_flags"]
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in ranked:
            row = {**p, "segments": _seg_str(p), "risk_flags": ",".join(p["risk_flags"]),
                   "themes": "; ".join(t["label"] for t in p["themes"]),
                   "categories": ", ".join(p["categories"])}
            w.writerow([row[c] for c in cols])
    md = [f"# Ranked promos ({meta['model']})\n"]
    for p in ranked:
        block = [f"## #{p['rank']}  Score {p['total_score']}  |  {_seg_str(p)}  |  {p['duration_sec']}s",
                 f"- **Scene:** {p['scene_summary']}  (emotion: {p['emotion']})",
                 f"- **Themes detected:** " + ("; ".join(f"{t['label']} (line {t['line']})" for t in p["themes"]) or "none"),
                 f"- **Categories:** {', '.join(p['categories']) or 'none'}",
                 f"- **Hook ({p['hook_type']}):** {p['hook']}",
                 f"  - script, first 3s: {p['hook_script_first_3s']}",
                 f"- **Cliffhanger ({p['cliffhanger_type']}):** {p['cliffhanger']}",
                 f"  - script, last lines: {p['cliffhanger_script_last_lines']}",
                 f"- **Why it works:** {p['why_it_works']}",
                 f"- Sub-scores: {p['scores']}  |  Platform risk: **{p['platform_risk']}** {p['risk_flags']}\n"]
        if p.get("cross_episode"):   # a join across episodes: show what the cold-viewer judge said
            block.insert(2, f"- **Cross-episode join:** coherence {p.get('coherence', '?')}/10 - {p.get('coherence_note', '')}")
        md += block
    (out / "promos.md").write_text("\n".join(md), encoding="utf-8")