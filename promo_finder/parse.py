"""Step 1 of the pipeline: plain code (no AI) turns a screenplay .txt into clean, timestamped data.

Analogy: cutting a film into labelled index cards, so everything later can point at a card by its id
instead of guessing timestamps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# One "beat" looks like:  **ROLE (emotion):**\nStart: (00:11.30) | End: (00:13.10)\ntext...
# The lookahead stops the text at the next beat, the next SCENE, the end marker, or end of file.
# Works for both layouts used in the pack (single-line and multi-line beats).
BEAT_RE = re.compile(
    r"\*\*([^\*:]+):\*\*\s*Start:\s*\((\d+:\d+\.\d+)\)\s*\|\s*End:\s*\((\d+:\d+\.\d+)\)\s*(.*?)"
    r"(?=\s*\*\*[^\*:]+:\*\*\s*Start:|\s*SCENE\s+\d+|\s*---\s*SCRIPT END|\Z)",
    re.DOTALL,
)

NON_DIALOGUE = {"ACTION": "action", "SOUND FX": "sound", "CAMERA": "camera"}


def to_seconds(ts: str) -> float:
    m, s = ts.strip().split(":")
    return int(m) * 60 + float(s)


def fmt_ts(sec: float) -> str:
    """Seconds -> 'MM:SS.ss' (same style as the screenplays)."""
    total = int(round(sec * 100))
    return f"{total // 6000:02d}:{(total % 6000) / 100:05.2f}"


@dataclass
class Line:
    id: str
    kind: str  # dialogue | action | sound
    speaker: str
    emotion: str
    start: float
    end: float
    text: str
    index: int = 0


@dataclass
class Camera:
    start: float
    end: float
    text: str


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    lines: list[Line]
    cameras: list[Camera]
    duration: float
    path: str = ""
    by_id: dict[str, Line] = field(default_factory=dict)

    def prompt_block(self) -> str:
        """Compact text the AI reads. Camera directions are dropped (they are only used for cut snapping)."""
        out = []
        for l in self.lines:
            who = l.speaker if not l.emotion else f"{l.speaker} ({l.emotion})"
            body = f'"{l.text}"' if l.kind == "dialogue" else l.text
            out.append(f"[{l.id} {fmt_ts(l.start)[:-1]}-{fmt_ts(l.end)[:-1]}] {who}: {body}")
        return "\n".join(out)


def parse_episode(text: str, path: str = "") -> Episode:
    num_m = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text)
    number = int(num_m.group(1)) if num_m else 0
    log_m = re.search(r"EPISODE LOGLINE:\s*(.*?)(?:CONTINUITY NOTES:|--- SCRIPT START ---)", text, re.DOTALL)
    cont_m = re.search(r"CONTINUITY NOTES:\s*(.*?)--- SCRIPT START ---", text, re.DOTALL)
    logline = " ".join(log_m.group(1).split()) if log_m else ""
    continuity = " ".join(cont_m.group(1).split()) if cont_m else ""

    raw = []
    cameras: list[Camera] = []
    for i, m in enumerate(BEAT_RE.finditer(text)):
        role_raw, s, e, body = m.groups()
        role = " ".join(role_raw.split())
        body = " ".join(body.split()).strip('"“” ')
        emo = re.search(r"\((.*?)\)", role)
        emotion = emo.group(1).strip() if emo else ""
        speaker = role.split("(")[0].strip()
        kind = NON_DIALOGUE.get(speaker.upper(), "dialogue")
        start, end = to_seconds(s), to_seconds(e)
        if kind == "camera":
            cameras.append(Camera(start, end, body))
        else:
            raw.append((start, i, Line("", kind, speaker, emotion, start, end, body)))

    raw.sort(key=lambda t: (t[0], t[1]))  # time order, file order breaks ties
    lines = []
    for n, (_, _, l) in enumerate(raw):
        l.id, l.index = f"L{n + 1:03d}", n
        lines.append(l)
    duration = max([l.end for l in lines] + [c.end for c in cameras] + [0.0])
    ep = Episode(number, logline, continuity, lines, cameras, duration, path)
    ep.by_id = {l.id: l for l in lines}
    return ep


def load_episodes(data_dir: str | Path) -> list[Episode]:
    eps = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        ep = parse_episode(p.read_text(encoding="utf-8", errors="ignore"), str(p))
        if ep.lines:
            eps.append(ep)
    eps.sort(key=lambda e: e.number)
    return eps
