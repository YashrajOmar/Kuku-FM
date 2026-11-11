"""Streamlit app: upload screenplays, find promos, filter by the 12 categories.
Run:  streamlit run app.py
Uploaded files are processed in a temporary folder and are never saved in the project.
"""
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.llm import PRESETS, LLMClient, MockClient
from promo_finder.pipeline import _seg_str, load_rubric, run, write_outputs

st.set_page_config(page_title="Kuku Promo Finder", layout="wide")
RC = load_rubric("rubric.yaml")
CATS = [spec["name"] for _, spec in sorted(RC["categories"].items())]

st.title("Kuku Promo Finder")
st.caption("AI reads each episode and proposes 20-60s promos. Code checks every timestamp, length and quote, "
           "and computes the score from rubric.yaml.")

with st.sidebar:
    st.header("Model (swappable)")
    provider = st.selectbox("Provider", [k for k in PRESETS if k != "custom"] + ["custom", "mock (offline demo)"])
    model = st.text_input("Model name (blank = provider default)")
    base_url = st.text_input("Base URL (only for 'custom')")
    key = st.text_input("API key (used for this run only, never stored)", type="password")
    stitch = st.checkbox("Also try cross-episode promos (1 extra AI call)", value=False)
    episodes = st.text_input("Episodes to process (e.g. 1-5, blank = all)")

files = st.file_uploader("Screenplays: .txt files or one .zip", type=["txt", "zip"], accept_multiple_files=True)

if st.button("Find promos", type="primary", disabled=not files):
    tmp = Path(tempfile.mkdtemp())
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.lower().endswith(".txt") and "__MACOSX" not in n:
                        (tmp / Path(n).name).write_bytes(z.read(n))
        else:
            (tmp / Path(f.name).name).write_bytes(f.getvalue())
    try:
        client = (MockClient() if provider.startswith("mock")
                  else LLMClient(provider, model or None, base_url=base_url or None, api_key=key or None))
        bar = st.progress(0.0, text="Starting")

        def cb(i, n, ep):
            bar.progress(i / n, text=f"Episode {ep} ({i}/{n})")

        ranked, meta = run(str(tmp), client, "rubric.yaml", episodes or None, verbose=False, stitch=stitch, progress=cb)
        out = tmp / "out"
        write_outputs(ranked, meta, str(out))
        st.session_state["result"] = {
            "ranked": ranked, "meta": meta,
            "files": {n: (out / n).read_bytes() for n in ("promos.csv", "promos.json", "promos.md", "run_meta.json")},
        }
        bar.empty()
    except Exception as e:
        st.error(f"Run failed: {e}")

# Results live in session_state, so changing a filter or clicking download does not erase them.
res = st.session_state.get("result")
if res:
    ranked, meta = res["ranked"], res["meta"]
    st.success(f"{len(ranked)} promos from {meta['episodes_processed']} episodes ({meta['model']}, "
               f"{meta['tokens_in']} tokens in, {meta['tokens_out']} out)")
    c1, c2, c3 = st.columns(3)
    sel = c1.multiselect("Filter by category", CATS)
    risk = c2.multiselect("Platform risk", ["low", "medium", "high"], default=["low", "medium", "high"])
    min_score = c3.slider("Minimum score", 0, 100, 0)
    show = [p for p in ranked
            if p["total_score"] >= min_score and p["platform_risk"] in risk
            and (not sel or set(sel) & set(p["categories"]))]

    rows = [{"rank": p["rank"], "score": p["total_score"], "segments": _seg_str(p), "sec": p["duration_sec"],
             "categories": ", ".join(p["categories"]), "hook": p["hook_type"], "cliffhanger": p["cliffhanger_type"],
             "risk": p["platform_risk"], "scene": p["scene_summary"]} for p in show]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    for p in show[:50]:
        with st.expander(f"#{p['rank']}  |  {p['total_score']}  |  {_seg_str(p)}  |  {', '.join(p['categories']) or 'no category'}"):
            st.markdown(f"**Scene:** {p['scene_summary']} (emotion: {p['emotion']})")
            st.markdown(f"**Hook ({p['hook_type']}):** {p['hook']}  \n*script, first 3s:* {p['hook_script_first_3s']}")
            st.markdown(f"**Cliffhanger ({p['cliffhanger_type']}):** {p['cliffhanger']}  \n*script, last lines:* {p['cliffhanger_script_last_lines']}")
            st.markdown(f"**Why it works:** {p['why_it_works']}")
            st.markdown("**Themes:** " + ("; ".join(f"{t['label']} (line {t['line']})" for t in p["themes"]) or "none"))
            st.markdown(f"**Platform risk:** {p['platform_risk']} {p['risk_flags']}  |  **Sub-scores:** {p['scores']}")
            st.dataframe(pd.DataFrame(p["transcript"], columns=["segment", "start", "end", "line"]), hide_index=True)

    with st.expander("Run details: rejected AI proposals, errors, episodes with no promo"):
        st.json({k: meta.get(k) for k in ("model", "episodes_processed", "promos_found", "promos_after_dedupe",
                                          "episodes_with_no_promo", "errors", "rejected_proposals", "stitch",
                                          "tokens_in", "tokens_out", "cached_calls", "seconds")})

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("promos.csv", res["files"]["promos.csv"], "promos.csv", "text/csv")
    d2.download_button("promos.json", res["files"]["promos.json"], "promos.json", "application/json")
    d3.download_button("promos.md", res["files"]["promos.md"], "promos.md", "text/markdown")
    d4.download_button("run_meta.json", res["files"]["run_meta.json"], "run_meta.json", "application/json")
