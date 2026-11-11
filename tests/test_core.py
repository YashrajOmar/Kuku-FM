"""Run: pytest -q   (needs the episodes in ./data)"""
import json
import re
from pathlib import Path

import pytest

from promo_finder.extract import build_system_prompt, find_promos, resolve_promo
from promo_finder.llm import MockClient, extract_json
from promo_finder.parse import fmt_ts, load_episodes
from promo_finder.pipeline import load_rubric, rank

RC = load_rubric()
EPS = load_episodes("data") if Path("data").exists() else []
needs_data = pytest.mark.skipif(not EPS, reason="episodes not in ./data")

# Every word you gave me must appear in some theme's "covers" text in rubric.yaml.
YOUR_WORDS = ["divorce", "crime", "sex", "rape", "fuck", "fingering", "masturbation", "kiss", "liplock", "undies",
              "naked", "nude", "bed", "eyes locked", "abusing", "bully", "harassment", "old with young",
              "same genders romanticized", "extremely expensive things", "beautiful nature scene",
              "having many partners at the same time", "asshole", "vagina", "breast", "pussy", "hit hard", "spank",
              "periods", "tension", "drugs", "weeds", "fire", "whore", "slap"]


def good(a, b, **kw):
    return {"segments": [{"start_line": a, "end_line": b}], "hook": "h", "hook_type": "cold_open", "cliffhanger": "c",
            "cliffhanger_type": "peril", "why_it_works": "w", "scene_summary": "s", "emotion": "tense",
            "themes": [], "scores": {"hook": 8, "cliffhanger": 8, "standalone": 8}, **kw}


def window(ep, secs):
    """Find a (start_id, end_id) whose span is about `secs` seconds."""
    a = ep.lines[0]
    b = next(l for l in ep.lines if l.end - a.start >= secs)
    return a.id, b.id


# ---------- rubric (no data needed) ----------
def test_every_user_word_is_in_the_theme_table():
    covers = " ".join(t["covers"].lower() for t in RC["themes"])
    missing = [w for w in YOUR_WORDS if w not in covers]
    assert not missing, f"words missing from rubric themes: {missing}"


def test_theme_table_is_well_formed():
    ids = [t["id"] for t in RC["themes"]]
    assert len(ids) == len(set(ids))
    for t in RC["themes"]:
        assert {"id", "label", "covers", "weight", "risk"} <= set(t)
        assert 0 <= t["weight"] <= 10 and t["risk"] in ("none", "low", "medium", "high")
        assert t.get("mode", "reward") in ("reward", "flag_only")
        if t.get("flag"):
            assert t["flag"] in RC["risk_flags"]


def test_prompt_lists_every_theme_and_has_no_stray_braces():
    s = build_system_prompt(RC)
    assert all(t["id"] in s for t in RC["themes"])
    assert '{"promos"' in s and "{p[" not in s


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"promos": []}\n```') == {"promos": []}


# ---------- parsing and validation (need data) ----------
@needs_data
def test_all_episodes_parse_and_ids_unique():
    assert len(EPS) == 50
    for ep in EPS:
        assert ep.lines and len(ep.by_id) == len(ep.lines)
        assert all(l.start <= l.end for l in ep.lines)


def test_fmt_ts_rounding():
    assert fmt_ts(59.999) == "01:00.00" and fmt_ts(71.3) == "01:11.30"


@needs_data
def test_valid_promo_resolves_with_real_script_text():
    ep = EPS[0]
    p, errs = resolve_promo(ep, good(*window(ep, 35)), RC)
    assert p and not errs and 20 <= p["duration_sec"] <= 60
    assert p["hook_script_first_3s"]            # copied from the script, not from the AI
    assert p["total_score"] == 64.0             # 8,8,8 and no themes: 10*(0.3*8 + 0.3*8 + 0.2*8 + 0.2*0)


@needs_data
def test_theme_adds_reward():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, themes=[{"id": "kiss", "line": a}]), RC)
    assert p["scores"]["audience_appeal"] == 7.0 and p["total_score"] == 78.0   # 64 + 10*0.2*7


@needs_data
def test_second_theme_adds_a_quarter():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, themes=[{"id": "kiss", "line": a}, {"id": "slap", "line": a}]), RC)
    assert p["scores"]["audience_appeal"] == 8.75   # 7 + 0.25*7


@needs_data
def test_theme_with_line_outside_clip_or_unknown_id_is_dropped():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, themes=[{"id": "kiss", "line": "L999"}, {"id": "not_a_theme", "line": a}]), RC)
    assert p["themes"] == [] and p["scores"]["audience_appeal"] == 0.0


@needs_data
def test_flag_only_theme_is_flagged_not_rewarded():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, themes=[{"id": "sexual_assault", "line": a}]), RC)
    assert p["scores"]["audience_appeal"] == 0.0
    assert p["platform_risk"] == "high" and "sexual_violence" in p["risk_flags"]


@needs_data
def test_theme_risk_overrides_a_low_ai_risk():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, platform_risk="low", themes=[{"id": "nudity", "line": a}]), RC)
    assert p["platform_risk"] == "high" and "nudity" in p["risk_flags"]


@needs_data
def test_minor_involved_is_never_rewarded():
    ep = EPS[0]
    a, b = window(ep, 35)
    p, _ = resolve_promo(ep, good(a, b, risk_flags=["minors_involved"], themes=[{"id": "kiss", "line": a}]), RC)
    assert p["scores"]["audience_appeal"] == 0.0 and p["platform_risk"] == "high"


@needs_data
@pytest.mark.parametrize("secs", [8, 90])
def test_length_rule_enforced(secs):
    ep = max(EPS, key=lambda e: e.duration)
    p, errs = resolve_promo(ep, good(*window(ep, secs)), RC)
    assert p is None and "outside" in errs[0]


@needs_data
def test_hallucinated_line_id_rejected():
    p, errs = resolve_promo(EPS[0], good("L999", "L1000"), RC)
    assert p is None and "unknown line id" in errs[0]


@needs_data
def test_reversed_and_overlapping_segments_rejected():
    ep = EPS[0]
    a, b = window(ep, 35)
    assert resolve_promo(ep, good(b, a), RC)[0] is None
    two = good(a, b)
    two["segments"] = [{"start_line": a, "end_line": b}, {"start_line": a, "end_line": b}]
    assert resolve_promo(ep, two, RC)[0] is None


@needs_data
def test_missing_scores_rejected():
    ep = EPS[0]
    bad = good(*window(ep, 35))
    del bad["scores"]["hook"]
    assert resolve_promo(ep, bad, RC)[0] is None


@needs_data
def test_scores_clamped_unknown_types_become_other_and_ai_cannot_set_appeal():
    ep = EPS[0]
    r = good(*window(ep, 35), platform_risk="high")
    r["scores"]["hook"] = 99
    r["scores"]["audience_appeal"] = 10      # must be ignored: appeal comes only from validated themes
    r["hook_type"] = "made_up"
    p, _ = resolve_promo(ep, r, RC)
    assert p["scores"]["hook"] == 10 and p["scores"]["audience_appeal"] == 0.0
    assert p["hook_type"] == "other" and p["platform_risk"] == "high"


@needs_data
def test_multi_segment_promo_adds_up():
    ep = max(EPS, key=lambda e: e.duration)
    a1, b1 = window(ep, 15)
    tail = [l for l in ep.lines if l.start >= ep.duration - 20]
    raw = good(a1, b1)
    raw["segments"].append({"start_line": tail[0].id, "end_line": tail[-1].id})
    p, errs = resolve_promo(ep, raw, RC)
    assert p and len(p["segments"]) == 2 and 20 <= p["duration_sec"] <= 60, errs


@needs_data
def test_retry_after_rejection():
    ep = EPS[0]
    replies = [json.dumps({"promos": [good("L999", "L998")]}), json.dumps({"promos": [good(*window(ep, 35))]})]

    class Flaky:
        label = "flaky"
        def complete(self, system, user):
            return replies.pop(0), {"prompt_tokens": 1, "completion_tokens": 1}

    promos, info = find_promos(ep, Flaky(), RC)
    assert len(promos) == 1 and info["attempts"] == 2 and info["rejected"]


@needs_data
def test_model_saying_nothing_is_accepted():
    class Empty:
        label = "empty"
        def complete(self, system, user):
            return '{"promos": []}', {}
    promos, info = find_promos(EPS[0], Empty(), RC)
    assert promos == [] and info["attempts"] == 1


@needs_data
def test_rank_dedupes_overlaps_and_orders():
    ep = EPS[0]
    a, b = window(ep, 35)
    p1, _ = resolve_promo(ep, good(a, b), RC)
    p2, _ = resolve_promo(ep, good(a, b, scores={"hook": 5, "cliffhanger": 5, "standalone": 5}), RC)
    ranked = rank([p2, p1])
    assert len(ranked) == 1 and ranked[0]["total_score"] == 64.0 and ranked[0]["rank"] == 1


@needs_data
def test_mock_end_to_end_all_episodes_respect_length_rule():
    from promo_finder.pipeline import run
    ranked, meta = run("data", MockClient(), verbose=False)
    assert ranked and all(20 <= p["duration_sec"] <= 60 for p in ranked)
    assert [p["rank"] for p in ranked] == list(range(1, len(ranked) + 1))
