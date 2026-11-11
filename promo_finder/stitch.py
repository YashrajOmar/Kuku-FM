"""Optional second pass: join moments from DIFFERENT episodes into one promo.

The first pass already found strong moments per episode. Here the AI sees only the best hook-moments and the best
cliffhanger-moments (a small pool, to keep the call cheap) and proposes 2-3 segment joins. Code then runs the same
checks as before: real episode and line ids, 20-60s total, scoring from rubric.yaml.
"""
from __future__ import annotations

from .extract import _say, resolve_promo
from .llm import extract_json
from .parse import fmt_ts

ADDENDUM = """
NEW TASK (this replaces the single-episode task above): you are given strong moments from DIFFERENT episodes of
the same show. Build promos by joining 2-3 segments, in the order they should play, usually from different episodes.
- Every segment is {"episode": <episode number>, "start_line": "L012", "end_line": "L020"}. Use only lines shown below.
- Segment 1 must be a killer hook. The last segment must end on a cliffhanger.
- The joined promo must make sense to a stranger who has seen none of the show. If the pieces only make sense with
  context from earlier episodes, do not join them.
- The total length of all segments must be 20-60 seconds; use the timestamps.
- Return an empty list if no join is genuinely good. Theme tags must cite line ids inside your chosen segments.
"""


JUDGE_SYSTEM = """You are an ordinary viewer who has NEVER seen this show. You are shown the transcript of one short promo clip.
Some parts were cut together from different scenes; "--- CUT ---" marks each cut. Judge ONLY from the transcript.
Ask yourself: do I understand who these people are and what is going on? Does the end feel like a natural, unresolved
continuation of the start, or do the pieces feel like unrelated scenes stuck together?
10 = one clear story. 7 = a stranger can follow it. 4 or less = different people, place or situation with no link.
Reply with ONLY this JSON: {"coherence": 0, "problem": "one sentence on what would confuse a stranger, or 'none'"}"""


def _gate(promo: dict, cfg: dict) -> str | None:
    """Cheap code checks for a join across episodes. Returns the reason to reject, or None if it passes."""
    eps = [s["episode"] for s in promo["segments"]]
    if len(set(eps)) < 2:
        return None
    if eps != sorted(eps):
        return "segments must play in episode order"
    gap = cfg.get("max_episode_gap", 5)
    if max(eps) - min(eps) > gap:
        return f"moments are {max(eps) - min(eps)} episodes apart (limit {gap})"
    return None


def judge_coherence(promo: dict, client) -> tuple[float, str, dict]:
    """A SEPARATE AI call that sees only the finished clip (no loglines, no notes), like a viewer on their phone."""
    lines, prev = [], None
    for seg, _a, _b, text in promo["transcript"]:
        if prev is not None and seg != prev:
            lines.append("--- CUT ---")
        lines.append(text)
        prev = seg
    reply, usage = client.complete(JUDGE_SYSTEM, "PROMO TRANSCRIPT:\n" + "\n".join(lines))
    try:
        d = extract_json(reply)
        return max(0.0, min(10.0, float(d["coherence"]))), str(d.get("problem", "")), usage
    except Exception:   # an unreadable verdict counts as a fail: never keep a join we could not check
        return 0.0, "judge reply was unreadable", usage


def _block(p: dict, by_num: dict) -> str:
    out = [f"=== SOURCE PROMO #{p['rank']} (hook {p['scores']['hook']:.0f}/10, cliffhanger {p['scores']['cliffhanger']:.0f}/10) ===",
           f"Summary: {p['scene_summary']}"]
    for s in p["segments"]:
        ep = by_num[s["episode"]]
        out.append(f"Episode {ep.number} logline: {ep.logline}")
        for l in ep.lines:
            if l.start >= s["start"] - 0.01 and l.end <= s["end"] + 0.01:
                out.append(f"[ep{ep.number} {l.id} {fmt_ts(l.start)[:-1]}-{fmt_ts(l.end)[:-1]}] {_say(l)}")
    return "\n".join(out)


def stitch_across_episodes(ranked: list[dict], eps: list, client, rc: dict, system: str):
    cfg = rc.get("stitch", {})
    pool, max_new = cfg.get("pool_size", 6), cfg.get("max_new_promos", 5)
    by_num = {e.number: e for e in eps}
    picked, seen = [], set()
    for p in (sorted(ranked, key=lambda x: -x["scores"]["hook"])[:pool]
              + sorted(ranked, key=lambda x: -x["scores"]["cliffhanger"])[:pool]):
        if p["rank"] not in seen:
            seen.add(p["rank"])
            picked.append(p)
    info = {"pool": sorted(seen), "rejected": [], "tokens_in": 0, "tokens_out": 0}
    if len({s["episode"] for p in picked for s in p["segments"]}) < 2:
        info["note"] = "need moments from at least 2 episodes"
        return [], info

    user = ("SOURCE MATERIAL (line ids repeat across episodes, so always name the episode):\n\n"
            + "\n\n".join(_block(p, by_num) for p in picked)
            + f"\n\nCreate up to {max_new} promos.\n")
    good: list[dict] = []
    for _ in range(2):
        text, usage = client.complete(system + ADDENDUM, user)
        info["tokens_in"] += usage.get("prompt_tokens", 0)
        info["tokens_out"] += usage.get("completion_tokens", 0)
        try:
            proposals = extract_json(text).get("promos", [])
        except Exception as e:
            info["rejected"].append(f"unparseable reply: {e}")
            user += "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object."
            continue
        problems = []
        for raw in proposals[: max_new + 2]:
            try:
                first = by_num.get(int(raw["segments"][0]["episode"]))
                promo, errs = resolve_promo(first, raw, rc, all_eps=by_num) if first else (None, ["unknown first episode"])
            except Exception as e:
                promo, errs = None, [f"bad proposal: {e}"]
            if promo:
                promo["cross_episode"] = len({s["episode"] for s in promo["segments"]}) > 1
                why = _gate(promo, cfg)
                if not why and promo["cross_episode"]:
                    score, note, u = judge_coherence(promo, client)
                    info["tokens_in"] += u.get("prompt_tokens", 0)
                    info["tokens_out"] += u.get("completion_tokens", 0)
                    promo["coherence"], promo["coherence_note"] = score, note
                    if score < cfg.get("min_coherence", 7):
                        why = f"coherence {score:.0f}/10: {note}"
                if why:
                    problems.append("join rejected: " + why)
                    info.setdefault("dropped_joins", []).append(why)
                else:
                    good.append(promo)
            else:
                problems += errs
        info["rejected"] += problems
        if good or not proposals:
            break
        user += "\n\nYour last answer was rejected: " + "; ".join(problems[:4]) + ". Fix this and answer again."
    return good[:max_new], info