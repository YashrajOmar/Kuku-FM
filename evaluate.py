"""Quality check: compare the tool's promos with YOUR hand-picked best moments.

  python evaluate.py template                                   -> writes eval/labels_template.csv
  python evaluate.py score outputs/run1/promos.json eval/labels.csv   -> prints a summary, writes eval/report.md

Labels: one row per episode you watched/read: episode, start, end (mm:ss.ss or seconds), note.
A pick counts as a HIT when it overlaps your label by at least half (intersection over union >= 0.5),
so a good pick does not have to match your timestamps to the hundredth of a second.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HIT_IOU = 0.5


def secs(x) -> float:
    x = str(x).strip()
    if ":" in x:
        m, s = x.split(":")
        return int(m) * 60 + float(s)
    return float(x)


def ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def best_match(label: dict, promos: list):
    """Best overlap between a label and any segment of any promo in the same episode."""
    best_v, best_p = 0.0, None
    for p in promos:
        for s in p["segments"]:
            if s["episode"] != label["episode"]:
                continue
            v = iou(label["start"], label["end"], s["start"], s["end"])
            if best_p is None or v > best_v:
                best_v, best_p = v, p
    return best_v, best_p


def template() -> None:
    Path("eval").mkdir(exist_ok=True)
    with open("eval/labels_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "start", "end", "note"])
        for ep in [1, 4, 8, 11, 15, 18, 22, 25, 29, 32, 36, 39, 43, 46, 50]:
            w.writerow([ep, "", "", ""])
    print("Wrote eval/labels_template.csv. Fill start/end for each episode, save as eval/labels.csv.")


def score(preds_path: str, labels_path: str) -> None:
    promos = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    labels = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("start", "")).strip() and str(r.get("end", "")).strip():
                labels.append({"episode": int(r["episode"]), "start": secs(r["start"]),
                               "end": secs(r["end"]), "note": r.get("note", "")})
    if not labels:
        sys.exit("No filled-in labels found.")

    rows = [(L, *best_match(L, promos)) for L in labels]
    hits = [r for r in rows if r[1] >= HIT_IOU]
    top_hits = 0
    for L in labels:
        in_ep = [p for p in promos if any(s["episode"] == L["episode"] for s in p["segments"])]
        if in_ep:
            top = min(in_ep, key=lambda p: p["rank"])
            if max(iou(L["start"], L["end"], s["start"], s["end"]) for s in top["segments"] if s["episode"] == L["episode"]) >= HIT_IOU:
                top_hits += 1

    # High-ranked picks in labelled episodes that overlap none of your labels: candidates for "where it picks badly"
    labelled = {L["episode"] for L in labels}
    suspicious = []
    for p in sorted(promos, key=lambda p: p["rank"])[:25]:
        eps = {s["episode"] for s in p["segments"]} & labelled
        if eps and all(iou(L["start"], L["end"], s["start"], s["end"]) < 0.2
                       for L in labels for s in p["segments"] if s["episode"] == L["episode"]):
            suspicious.append(p)

    n = len(labels)
    lines = ["# Evaluation report", "",
             f"- Labelled episodes: {n}",
             f"- Hit rate (any promo overlaps your pick, IoU >= {HIT_IOU}): {len(hits)}/{n}",
             f"- Top-ranked promo in the episode is a hit: {top_hits}/{n}",
             f"- Median best overlap: {statistics.median(r[1] for r in rows):.2f}", "",
             "## Misses: you picked this, the tool did not", ""]
    for L, v, p in rows:
        if v < HIT_IOU:
            near = f"closest tool pick: {_seg(p)} (overlap {v:.2f}): {p['why_it_works']}" if p else "tool found nothing in this episode"
            lines.append(f"- Ep{L['episode']} {ts(L['start'])}-{ts(L['end'])} {L['note']}  ->  {near}")
    lines += ["", "## Possible bad picks: ranked high by the tool, overlap none of your labels (review by hand)", ""]
    for p in suspicious:
        lines.append(f"- #{p['rank']} {_seg(p)} score {p['total_score']}: {p['why_it_works']}")
    Path("eval").mkdir(exist_ok=True)
    Path("eval/report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print("Full report: eval/report.md")


def _seg(p: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {ts(s['start'])}-{ts(s['end'])}" for s in p["segments"])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "template":
        template()
    elif len(sys.argv) == 4 and sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
