"""Tests for the 12-category layer, cross-episode stitching and the overlap-based eval. Run: pytest -q"""
import json
from pathlib import Path

import pytest

import evaluate as ev
from promo_finder.extract import resolve_promo, theme_categories
from promo_finder.parse import load_episodes
from promo_finder.pipeline import load_rubric, rank
from promo_finder.stitch import stitch_across_episodes

RC = load_rubric()
EPS = load_episodes("data") if Path("data").exists() else []
needs_data = pytest.mark.skipif(len(EPS) < 2, reason="episodes not in ./data")


def window(ep, secs):
    a = ep.lines[0]
    b = next(l for l in ep.lines if l.end - a.start >= secs)
    return a.id, b.id


def promo_raw(ep, secs=35, themes=()):
    a, b = window(ep, secs)
    return {"segments": [{"start_line": a, "end_line": b}], "hook": "h", "hook_type": "cold_open", "cliffhanger": "c",
            "cliffhanger_type": "peril", "why_it_works": "w", "scene_summary": "s", "emotion": "tense",
            "themes": [{"id": t, "line": a} for t in themes],
            "scores": {"hook": 8, "cliffhanger": 8, "standalone": 8}}


def test_every_theme_is_in_exactly_one_category_and_all_12_exist():
    assert sorted(RC["categories"]) == list(range(1, 13))
    listed = [t for spec in RC["categories"].values() for t in spec["themes"]]
    ids = [t["id"] for t in RC["themes"]]
    assert sorted(listed) == sorted(ids), "every theme must be in exactly one category"


def test_theme_categories_lookup():
    cats = theme_categories(RC)
    assert cats["kiss"] == "Sexual / vulgar" and cats["slap"] == "Violence" and cats["fire_peril"] == "Danger"


@needs_data
def test_promo_carries_categories():
    p, errs = resolve_promo(EPS[0], promo_raw(EPS[0], themes=["kiss", "slap"]), RC)
    assert p and not errs
    assert p["categories"] == ["Sexual / vulgar", "Violence"]


@needs_data
def test_cross_episode_segments_resolve_and_unknown_episode_is_rejected():
    e0, e1 = EPS[0], EPS[1]
    a0, b0 = window(e0, 12)
    a1, b1 = window(e1, 12)
    raw = promo_raw(e0)
    raw["segments"] = [{"episode": e0.number, "start_line": a0, "end_line": b0},
                       {"episode": e1.number, "start_line": a1, "end_line": b1}]
    p, errs = resolve_promo(e0, raw, RC, all_eps={e.number: e for e in EPS})
    assert p and [s["episode"] for s in p["segments"]] == [e0.number, e1.number], errs
    raw["segments"][1]["episode"] = 9999
    assert resolve_promo(e0, raw, RC, all_eps={e.number: e for e in EPS})[0] is None


@needs_data
def test_stitch_pass_with_fake_model():
    e0, e1 = EPS[0], EPS[1]
    r0 = resolve_promo(e0, promo_raw(e0), RC)[0]
    r1 = resolve_promo(e1, promo_raw(e1), RC)[0]
    ranked = rank([r0, r1])
    a0, b0 = window(e0, 12)
    a1, b1 = window(e1, 12)
    cross = promo_raw(e0)
    cross["segments"] = [{"episode": e0.number, "start_line": a0, "end_line": b0},
                         {"episode": e1.number, "start_line": a1, "end_line": b1}]

    # class Fake:
    #     label = "fake"
    #     def complete(self, system, user):
    #         assert "SOURCE MATERIAL" in user
    #         return json.dumps({"promos": [cross]}), {"prompt_tokens": 1, "completion_tokens": 1}

    class Fake:
        label = "fake"
        def complete(self, system, user):
            if "PROMO TRANSCRIPT:" in user:
                return '{"coherence": 9, "problem": "none"}', {"prompt_tokens": 1, "completion_tokens": 1}
            assert "SOURCE MATERIAL" in user
            return json.dumps({"promos": [cross]}), {"prompt_tokens": 1, "completion_tokens": 1}

    out, info = stitch_across_episodes(ranked, [e0, e1], Fake(), RC, "system")
    assert len(out) == 1 and out[0]["cross_episode"] is True


def test_eval_overlap_maths():
    assert ev.secs("01:05.5") == 65.5 and ev.secs("12") == 12.0
    assert ev.iou(0, 10, 5, 15) == pytest.approx(5 / 15)
    assert ev.iou(0, 10, 20, 30) == 0.0
    assert ev.iou(10, 40, 12, 38) > 0.5    # a near-miss on timestamps is still a hit
