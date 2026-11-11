"""Swappable model layer.

Nearly every provider (Gemini, Groq, OpenRouter, OpenAI, local Ollama...) speaks the same "OpenAI chat" protocol.
So swapping the model = changing three values: base_url, model, api key. No code changes.
Analogy: a universal power adapter: the pipeline is the appliance, the provider is the wall socket.

Keys are read from environment variables (or a .env file). Never paste a key into code or chat.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

# Defaults only. Model names change often, so override with --model or LLM_MODEL.
PRESETS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-flash-latest", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "custom": (os.getenv("LLM_BASE_URL", ""), os.getenv("LLM_MODEL", ""), "LLM_API_KEY"),
}


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```fences``` or add chatter. Pull out the outermost {...}."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[a : b + 1])


class QuotaExhausted(RuntimeError):
    """The provider says the DAILY allowance is used up. Waiting a few seconds will not help, so we stop at once."""


def _is_daily_limit(msg: str) -> bool:
    m = msg.lower()
    return "per day" in m or "perday" in m or "(tpd)" in m or "(rpd)" in m


class LLMClient:
    def __init__(self, provider: str = "gemini", model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, cache_dir: str | None = ".cache", temperature: float = 0.2):
        load_dotenv()
        if provider not in PRESETS:
            raise ValueError(f"unknown provider {provider!r}; choose from {list(PRESETS)}")
        p_url, p_model, key_env = PRESETS[provider]
        self.provider = provider
        custom = provider == "custom"  # LLM_* variables belong to the custom provider only, so they never leak into groq/gemini runs
        self.base_url = base_url or (os.getenv("LLM_BASE_URL") if custom else None) or p_url
        self.model = model or (os.getenv("LLM_MODEL") if custom else None) or p_model
        self.api_key = api_key or os.getenv(key_env) or (os.getenv("LLM_API_KEY") if custom else None)
        if not self.api_key:
            raise RuntimeError(f"No API key. Set {key_env} in your environment or .env file.")
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        from openai import OpenAI  # imported here so the mock/offline path needs no SDK

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def _key(self, system: str, user: str) -> Path:
        h = hashlib.sha256(f"{self.model}\n{self.temperature}\n{system}\n{user}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        """Returns (reply_text, usage). Cached on disk so re-runs are free and reproducible."""
        if self.cache_dir:
            f = self._key(system, user)
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                return d["text"], {**d["usage"], "cached": True}
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last = None
        for attempt in range(6):  # free tiers rate-limit (HTTP 429): back off and retry
            try:
                try:
                    r = self._client.chat.completions.create(
                        model=self.model, messages=msgs, temperature=self.temperature,
                        response_format={"type": "json_object"})
                except Exception as e:  # some providers reject response_format; retry without it
                    if "response_format" in str(e).lower() or "json" in str(e).lower():
                        r = self._client.chat.completions.create(
                            model=self.model, messages=msgs, temperature=self.temperature)
                    else:
                        raise
                text = r.choices[0].message.content or ""
                u = getattr(r, "usage", None)
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                         "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
                if self.cache_dir:
                    self._key(system, user).write_text(json.dumps({"text": text, "usage": usage}), encoding="utf-8")
                return text, {**usage, "cached": False}
            except Exception as e:
                last = e
                code = getattr(e, "status_code", None)
                if code in (400, 401, 403, 404, 413, 422):  # retrying cannot fix a wrong key, model name, URL or too-big request: fail fast
                    raise RuntimeError(f"HTTP {code} for model '{self.model}' at {self.base_url} -> {e}") from e
                if code == 429 and _is_daily_limit(str(e)):  # per-minute limits are worth waiting for; the daily one is not
                    raise QuotaExhausted(f"daily limit reached for '{self.model}' -> {e}") from e
                time.sleep(min(60, 4 * 2**attempt))
        raise RuntimeError(f"LLM call failed after retries: {last}")

    def list_models(self) -> list[str]:
        """Model ids this provider and key can use. Handy when a model name has been retired (HTTP 404)."""
        return sorted(m.id for m in self._client.models.list())


class MockClient:
    """Offline stand-in used ONLY to test the plumbing (parsing, validation, ranking, output files).
    It reads the line ids out of the prompt and returns valid-looking JSON with made-up scores.
    Its picks are NOT a quality baseline."""

    provider, model = "mock", "mock-v0"
    label = "mock:mock-v0"

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        rows = re.findall(r"\[(L\d+) (\d+):(\d+\.\d)-(\d+):(\d+\.\d)\]", user)
        L = [(i, int(a) * 60 + float(b), int(c) * 60 + float(d)) for i, a, b, c, d in rows]
        promos = []
        if L:
            end = L[-1]
            first = next(x for x in L if x[2] >= end[2] - 36)
            promos.append(self._p(first[0], end[0], 6, 8))
            head = [x for x in L if x[2] <= 30]
            if len(head) >= 3:
                promos.append(self._p(head[0][0], head[-1][0], 7, 5))
        return json.dumps({"promos": promos}), {"prompt_tokens": len(user) // 4, "completion_tokens": 200, "cached": False}

    @staticmethod
    def _p(a, b, hook, cliff):
        return {"segments": [{"start_line": a, "end_line": b}], "hook": "mock hook", "hook_type": "cold_open",
                "cliffhanger": "mock cliffhanger", "cliffhanger_type": "unresolved_confrontation",
                "why_it_works": "mock", "scores": {"hook": hook, "cliffhanger": cliff, "standalone": 5, "audience_appeal": 5},
                "platform_risk": "low", "risk_flags": []}