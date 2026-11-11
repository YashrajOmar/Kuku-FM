"""AI does the judgment, code does the guarantees.

The AI never writes timestamps. It only points at line ids ("L012" to "L031"). Code then:
  - looks up the real start/end times from the screenplay (so timestamps can't be hallucinated),
  - snaps the cut to shot boundaries,
  - enforces the 20-60 second rule, ordering and overlap,
  - copies the hook / cliffhanger text straight from the script (so quotes are real),
  - keeps only the audience themes the AI backed with a real line id inside the clip,
  - computes audience appeal, risk and the final score from rubric.yaml.

There is NO keyword list anywhere in the code. Themes are decided by the AI from the meaning of the scene.
"""
from __future__ import annotations

from .llm import extract_json
from .parse import Episode, fmt_ts

SNAP_SECONDS = 2.0  # max distance we will move a cut to land on a shot boundary
RISK_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
RISK_NAME = {0: "low", 1: "low", 2: "medium", 3: "high"}


def build_system_prompt(rc: dict) -> str:
    p = rc["promo"]
    hooks = "\n".join(f"  - {k}: {v}" for k, v in rc["hook_types"].items())
    cliffs = "\n".join(f"  - {k}: {v}" for k, v in rc["cliffhanger_types"].items())
    principles = "\n".join(f"  - {s}" for s in rc["hook_principles"])
    examples = "\n".join(f'  - "{s}"' for s in rc["cliffhanger_examples"])
    themes = "\n".join(f"  - {t['id']}: {t['label']} (covers: {t['covers']})" for t in rc["themes"])
    flags = ", ".join(rc["risk_flags"])
    raw = f"""\
    You are a senior promo editor for a vertical microdrama app (ReelShort / DramaBox style).
    You get the timestamped screenplay of ONE episode: every beat has a line id, a time range, a speaker, an emotion and text.
    Choose the best promo clips: {p['min_seconds']}-{p['max_seconds']} second ads for Instagram / TikTok / YouTube that make a
    scrolling stranger want to watch the show.

    A great promo:
    1. HOOK: the FIRST line of the clip must stop the scroll within {p['hook_window_seconds']} seconds. Do not start with set-up or
       quiet context; start on the shocking line, slap, reveal or danger. Hook principles:
    {principles}
       Hook types:
    {hooks}
    2. CLIFFHANGER: the LAST line must be an unresolved moment so the viewer needs the next episode. Never include
       the resolution. Cliffhanger types:
    {cliffs}
       The feel we want:
    {examples}
    3. STANDALONE: makes sense to someone who has never seen the show. Judge this from the clip ALONE. The logline and
       continuity notes are only there to help you understand the story; the viewer will not see them.
    4. AUDIENCE THEMES: our audience is Gen Z and clicks for the themes below. First understand the scene: what is
       happening, who holds the power, how each character feels. Then list every theme the clip GENUINELY shows.
       Decide by the MEANING of the scene, never by a single word. A garden full of "weeds" is not drugs, and "the
       room was on fire with gossip" is not fire. Each theme must cite the line id that shows it, and that line must
       be inside your clip. Do not tag a theme you cannot point to. Themes:
    {themes}

    How to choose:
    - Understand the emotion and stakes of each scene, then choose the strongest {p['min_seconds']}-{p['max_seconds']}s stretch.
    - A promo is one continuous stretch, OR up to 3 segments joined in order (for example a cold-open hook, then
      the final cliffhanger). Total length of all segments must be {p['min_seconds']}-{p['max_seconds']} seconds; use the timestamps.
    - Return 1 to {p['max_promos_per_episode']} promos. Return fewer, or an empty list, if the episode has no genuinely strong moment.
      Different promos must not overlap.
    - segments use line ids from the script (start_line and end_line are both included).

    Scoring: integers 0-10 for hook, cliffhanger and standalone ONLY (themes are scored by code, so do not score them).
    Be harsh and spread your scores. 9-10 = would stop most scrollers, 7-8 = strong, 5-6 = average, below 5 = weak.
    Most promos should land between 4 and 8.
    Also set platform_risk (low / medium / high) = how likely an ad platform is to restrict this clip, and risk_flags
    chosen from: {flags}. If any character who is a minor appears in romantic, sexual or violent content, add
    minors_involved. These are information flags for the human editor; do not let them lower your other scores.

    Reply with ONLY this JSON, no other text:
    {{"promos": [{{
      "segments": [{{"start_line": "L012", "end_line": "L031"}}],
      "scene_summary": "2 sentences: what is happening, who is involved, what is at stake",
      "emotion": "dominant emotion of the clip, 2-3 words",
      "hook": "one sentence: what happens in the first 3 seconds",
      "hook_type": "one of the hook types",
      "cliffhanger": "one sentence: what is left unresolved at the cut",
      "cliffhanger_type": "one of the cliffhanger types",
      "why_it_works": "one sentence",
      "themes": [{{"id": "theme id from the list", "line": "L014"}}],
      "scores": {{"hook": 0, "cliffhanger": 0, "standalone": 0}},
      "platform_risk": "low",
      "risk_flags": []
    }}]}}
    """
    # strip the 4-space source indent (textwrap.dedent would be thrown off by the inserted bullet lists)
    return "\n".join(l[4:] if l.startswith("    ") else l for l in raw.split("\n"))


def build_user_prompt(ep: Episode) -> str:
    return (f"EPISODE {ep.number}  (length {fmt_ts(ep.duration)[:-1]})\n"
            f"LOGLINE: {ep.logline}\nCONTINUITY: {ep.continuity}\n\n"
            f"SCRIPT: one beat per line, [id start-end] speaker: text\n{ep.prompt_block()}\n")


def _snap(ep: Episode, a, b) -> tuple[float, float]:
    """Move the cut to the nearest shot boundary (within SNAP_SECONDS) so we don't cut mid-shot."""
    start, end = a.start, b.end
    for c in ep.cameras:
        if c.start <= a.start < c.end and a.start - c.start <= SNAP_SECONDS:
            start = min(start, c.start)
        if c.start < b.end <= c.end and c.end - b.end <= SNAP_SECONDS:
            end = max(end, c.end)
    return max(0.0, start), min(ep.duration, end)


TRIM_MAX_SECONDS = 5.0   # a clip up to this much over the limit is trimmed instead of rejected


def _build_segments(ep: Episode, segs_raw: list, p: dict, all_eps: dict | None, snap: bool) -> tuple[list, list]:
    """Turn the AI's line ids into real segments. snap=True moves cuts to shot boundaries; False uses exact line times."""
    errs, segs, prev_end, prev_ep = [], [], -1.0, None
    for s in segs_raw:
        src = _source_episode(ep, s, all_eps)   # which episode this segment comes from
        if src is None:
            errs.append(f"unknown episode in {s}")
            continue
        a, b = src.by_id.get(str(s.get("start_line"))), src.by_id.get(str(s.get("end_line")))
        if not a or not b:
            errs.append(f"unknown line id in {s}")
            continue
        if a.index > b.index:
            errs.append(f"start_line after end_line in {s}")
            continue
        start, end = _snap(src, a, b) if snap else (a.start, b.end)
        if src.number == prev_ep and start < prev_end:
            errs.append("segments overlap or are out of order")
            continue
        if end - start < p["min_segment_seconds"]:
            errs.append(f"segment {fmt_ts(start)}-{fmt_ts(end)} shorter than {p['min_segment_seconds']}s")
            continue
        prev_end, prev_ep = end, src.number
        inside = [l for l in src.lines if l.start >= start - 0.01 and l.end <= end + 0.01]
        segs.append({"episode": src.number, "start": round(start, 2), "end": round(end, 2),
                     "start_ts": fmt_ts(start), "end_ts": fmt_ts(end), "start_line": a.id, "end_line": b.id,
                     "lines": inside})
    return segs, errs


def resolve_promo(ep: Episode, raw: dict, rc: dict, all_eps: dict | None = None) -> tuple[dict | None, list[str]]:
    """Turn one AI-proposed promo into a verified promo (or return the reasons it was rejected)."""
    p = rc["promo"]
    segs_raw = raw.get("segments")
    if not isinstance(segs_raw, list) or not segs_raw:
        return None, ["no segments"]
    segs, errs = _build_segments(ep, segs_raw, p, all_eps, snap=True)
    if errs or not segs:
        return None, errs or ["no valid segments"]
    total = round(sum(s["end"] - s["start"] for s in segs), 2)
    trimmed = False
    if total > p["max_seconds"]:
        # Fallback 1: snapping to shot boundaries may have pushed a good clip over the limit. Try the exact line times.
        exact, exact_errs = _build_segments(ep, segs_raw, p, all_eps, snap=False)
        if exact and not exact_errs:
            exact_total = round(sum(s["end"] - s["start"] for s in exact), 2)
            if exact_total < total:
                segs, total = exact, exact_total
    if p["max_seconds"] < total <= p["max_seconds"] + TRIM_MAX_SECONDS:
        # Fallback 2: only a little too long. Drop the last line(s) instead of throwing the whole clip away.
        last = segs[-1]
        keep = [l for l in last["lines"] if l.end <= last["end"] - (total - p["max_seconds"]) + 0.01]
        if keep and keep[-1].end - last["start"] >= p["min_segment_seconds"]:
            last.update({"end": round(keep[-1].end, 2), "end_ts": fmt_ts(keep[-1].end), "end_line": keep[-1].id, "lines": keep})
            total, trimmed = round(sum(s["end"] - s["start"] for s in segs), 2), True
    if total < p["min_seconds"] or total > p["max_seconds"]:
        return None, [f"total length {total}s is outside {p['min_seconds']}-{p['max_seconds']}s "
                      f"(adjust start_line/end_line so segments add up to {p['min_seconds']}-{p['max_seconds']}s)"]

    # AI sub-scores (everything in weights except audience_appeal, which code computes from themes)
    sc = raw.get("scores") or {}
    ai_keys = [k for k in rc["weights"] if k != "audience_appeal"]
    try:
        sub = {k: max(0.0, min(10.0, float(sc[k]))) for k in ai_keys}
    except (KeyError, TypeError, ValueError):
        return None, [f"scores must include numbers for {ai_keys}"]
    if trimmed:   # the AI scored and described a different ending, so be conservative about the cliffhanger
        sub["cliffhanger"] = max(0.0, sub["cliffhanger"] - 1.0)

    # Themes: keep only valid theme ids that cite a line which really is inside the clip
    tax = {t["id"]: t for t in rc["themes"]}
    cat_of = theme_categories(rc)
    inside_ids = {l.id for s in segs for l in s["lines"]}
    themes: list[dict] = []
    for t in raw.get("themes") or []:
        if not isinstance(t, dict):
            continue
        tid, ln = str(t.get("id")), str(t.get("line"))
        if tid in tax and ln in inside_ids and all(x["id"] != tid for x in themes):
            themes.append({"id": tid, "label": tax[tid]["label"], "category": cat_of.get(tid, ""), "line": ln})

    # Audience appeal: strongest rewarded theme + 25% of the next TWO strongest, capped at 10.
    # (Only two extras count, so tagging many weak themes cannot beat one strong theme.)
    ws = sorted((tax[t["id"]]["weight"] for t in themes if tax[t["id"]].get("mode", "reward") == "reward"), reverse=True)
    appeal = min(10.0, ws[0] + 0.25 * sum(ws[1:3])) if ws else 0.0

    # Platform risk = the worst of what the AI said and what the detected themes imply
    flags = {f for f in (raw.get("risk_flags") or []) if f in rc["risk_flags"]}
    flags |= {tax[t["id"]]["flag"] for t in themes if tax[t["id"]].get("flag")}
    level = max([RISK_RANK.get(str(raw.get("platform_risk", "low")).lower(), 2)]
                + [RISK_RANK.get(tax[t["id"]].get("risk", "none"), 0) for t in themes])
    if "minors_involved" in flags:   # a minor in romantic / sexual / violent content is never rewarded
        level, appeal = 3, 0.0
    risk = RISK_NAME[level]

    sub["audience_appeal"] = round(appeal, 2)
    total_score = 10 * sum(rc["weights"][k] * sub[k] for k in rc["weights"]) - rc["platform_risk_penalty"].get(risk, 0)

    first, last = segs[0], segs[-1]
    hook_lines = [l for l in first["lines"] if l.start < first["start"] + p["hook_window_seconds"]]
    return {
        "episode": ep.number,
        "segments": [{k: v for k, v in s.items() if k != "lines"} for s in segs],
        "duration_sec": total,
        "trimmed_to_fit": trimmed,   # True when the last line(s) were dropped to get under the length limit
        "scene_summary": str(raw.get("scene_summary", "")),
        "emotion": str(raw.get("emotion", "")),
        "themes": themes,
        "categories": sorted({t["category"] for t in themes if t["category"]}),
        "hook": str(raw.get("hook", "")),
        "hook_type": _pick(raw.get("hook_type"), rc["hook_types"]),
        "hook_script_first_3s": " | ".join(_say(l) for l in hook_lines),          # copied from the script
        "cliffhanger": (f"[clip trimmed to fit 60s] now ends on: {_say(last['lines'][-1])}" if trimmed and last["lines"]
                        else str(raw.get("cliffhanger", ""))),   # after a trim, the AI's own description no longer matches the ending
        "cliffhanger_type": _pick(raw.get("cliffhanger_type"), rc["cliffhanger_types"]),
        "cliffhanger_script_last_lines": " | ".join(_say(l) for l in last["lines"][-2:]),  # copied from the script
        "why_it_works": str(raw.get("why_it_works", "")),
        "scores": sub,
        "total_score": round(total_score, 1),
        "platform_risk": risk,
        "risk_flags": sorted(flags),
        "transcript": [[f"Seg{i + 1}", fmt_ts(l.start), fmt_ts(l.end), _say(l)] for i, s in enumerate(segs) for l in s["lines"]],
    }, []


def theme_categories(rc: dict) -> dict:
    """theme id -> category name, read from the `categories` block of rubric.yaml."""
    return {tid: spec["name"] for spec in rc.get("categories", {}).values() for tid in spec["themes"]}


def _source_episode(ep: Episode, seg: dict, all_eps: dict | None):
    """Episode a segment comes from. Single-episode promos never set it, so they stay in `ep`.
    With all_eps (cross-episode mode) the segment must name a real episode, otherwise None."""
    if all_eps is not None and seg.get("episode") is not None:
        try:
            return all_eps.get(int(seg["episode"]))
        except (TypeError, ValueError):
            return None
    return ep


def _say(l) -> str:
    return f'{l.speaker}: "{l.text}"' if l.kind == "dialogue" else f"[{l.speaker}] {l.text}"


def _pick(v, allowed: dict) -> str:
    return v if v in allowed else "other"


def find_promos(ep: Episode, client, rc: dict, system: str | None = None) -> tuple[list[dict], dict]:
    """One AI call per episode (plus at most one retry if every proposal was rejected)."""
    system = system or build_system_prompt(rc)
    user = build_user_prompt(ep)
    info = {"episode": ep.number, "attempts": 0, "rejected": [], "tokens_in": 0, "tokens_out": 0, "cached": 0}
    for attempt in range(2):
        info["attempts"] += 1
        text, usage = client.complete(system, user)
        info["tokens_in"] += usage.get("prompt_tokens", 0)
        info["tokens_out"] += usage.get("completion_tokens", 0)
        info["cached"] += int(bool(usage.get("cached")))
        try:
            proposals = extract_json(text).get("promos", [])
        except Exception as e:
            info["rejected"].append(f"unparseable reply: {e}")
            user += "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object."
            continue
        good, problems = [], []
        for raw in proposals[: rc["promo"]["max_promos_per_episode"] + 2]:
            promo, errs = resolve_promo(ep, raw, rc)
            (good.append(promo) if promo else problems.extend(errs))
        info["rejected"] += problems
        if good or not proposals:  # accepted something, or the model honestly said "nothing strong here"
            return good, info
        user += "\n\nYour last answer was rejected: " + "; ".join(problems[:4]) + ". Fix this and answer again."
    return [], info