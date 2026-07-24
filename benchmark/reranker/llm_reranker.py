#!/usr/bin/env python3
"""
llm_reranker.py – Listwise LLM Reranker (Stage 2.5, Datei-Ebene).

Statt jeden Chunk einzeln durch einen kleinen Cross-Encoder zu schicken, bekommt
ein LLM die N Stage-1-Kandidat-Dateien (je mit einer kompakten Symbol-Übersicht)
plus den Issue-Text und wählt in EINEM Call die wahrscheinlichsten Dateien aus,
die geändert werden müssen.

Warum das für Feature-Location besser passt als ein 560M-Cross-Encoder:
  - Das LLM kann Issue → Code semantisch "durchdenken" (Bug-Lokalisierung),
    nicht nur Text-Ähnlichkeit messen.
  - Es arbeitet auf DATEI-Ebene (= die Eval-Einheit), nicht auf 1600 Chunks.
  - Ein Call pro Query statt 1600 Forward-Passes → schnell, nicht MPS-gebunden.

Nutzt denselben OpenAI-kompatiblen Endpunkt wie precompute_llm_expansions.py
(LM Studio, Default http://localhost:1234/v1). Verträgt Qwen3-Thinking-Modelle.
"""

from __future__ import annotations

import re
import json
import time
from typing import List, Tuple, Optional


class LLMListwiseReranker:
    def __init__(
        self,
        api_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        timeout: float = 180.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        issue_max_chars: int = 1500,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.issue_max_chars = issue_max_chars
        self.max_retries = max_retries
        self.verbose = verbose
        # Diagnostik
        self.parse_failures = 0
        self.calls = 0

    # ── öffentliche API ───────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[str]:
        """Rankt Kandidat-Dateien per LLM.

        candidates: [(file_path, outline_str), …] in Stage-1-Reihenfolge.
        Rückgabe: gerankte file_paths (LLM-Auswahl zuerst, dann mit der
        Stage-1-Reihenfolge aufgefüllt, damit immer top_k Slots belegt sind).
        """
        if not candidates:
            return []

        prompt = self._build_prompt(query, candidates, top_k)
        text = self._call(prompt)
        order = self._parse_ids(text, n=len(candidates)) if text else []
        if not order:
            self.parse_failures += 1

        ranked = [candidates[i][0] for i in order]
        # Mit Stage-1-Reihenfolge auffüllen (nie schlechter als "kein Rerank").
        seen = set(ranked)
        for fp, _ in candidates:
            if fp not in seen:
                ranked.append(fp)
                seen.add(fp)
        return ranked[:top_k] if top_k else ranked

    # ── Prompt & Parsing ──────────────────────────────────────

    def _build_prompt(self, query: str, candidates, top_k) -> str:
        issue = (query or "")[: self.issue_max_chars]
        lines = []
        for i, (fp, outline) in enumerate(candidates, start=1):
            outline = (outline or "").strip()
            lines.append(f"{i}: {fp} — {outline}" if outline else f"{i}: {fp}")
        catalog = "\n".join(lines)
        return (
            "You are an expert software engineer doing bug/feature localization. "
            "Given an issue and a list of candidate source files (with their key "
            "classes/functions), identify which files most likely need to be "
            "changed to resolve the issue.\n\n"
            f"ISSUE:\n{issue}\n\n"
            f"CANDIDATE FILES:\n{catalog}\n\n"
            f"Return ONLY a JSON array of the file ids most likely to require "
            f"changes, best first, at most {top_k} ids. Example: [3, 1, 7]\n"
            "Do not include any text outside the JSON array."
        )

    def _parse_ids(self, text: str, n: int) -> List[int]:
        """Extrahiert 1-basierte IDs aus der LLM-Antwort → 0-basierte Indizes."""
        blocks = re.findall(r"\[[^\[\]]*\]", text, flags=re.DOTALL)
        raw: List[int] = []
        if blocks:
            block = blocks[-1]
            try:
                raw = [int(x) for x in json.loads(block)]
            except Exception:
                raw = [int(x) for x in re.findall(r"\d+", block)]
        else:
            raw = [int(x) for x in re.findall(r"\d+", text)]

        out, seen = [], set()
        for i in raw:
            j = i - 1  # 1-based → 0-based
            if 0 <= j < n and j not in seen:
                out.append(j)
                seen.add(j)
        return out

    # ── LLM-Call (LM Studio, wie precompute_llm_expansions.py) ─

    def _call(self, prompt: str) -> Optional[str]:
        import requests

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.calls += 1
                resp = requests.post(
                    f"{self.api_url}/chat/completions",
                    json=body,
                    timeout=self.timeout,
                    headers={"Connection": "close"},
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                message = choice.get("message", {})
                content = (message.get("content") or "").strip()
                if not content:
                    # Qwen3-Thinking: Antwort evtl. in reasoning_content.
                    content = (message.get("reasoning_content") or "").strip()
                if content:
                    return content
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)
            except Exception as e:  # Timeout/Connection/HTTP
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)
        if self.verbose and last_error:
            print(f"    [LLM] Call fehlgeschlagen: {last_error}")
        return None

    # ── Health-Check ──────────────────────────────────────────

    def ping(self) -> bool:
        import requests
        try:
            resp = requests.get(f"{self.api_url}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            mid = models[0].get("id", "?") if models else "?"
            if self.verbose:
                print(f"  [LLM] LM Studio erreichbar, Modell: {mid}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"  [LLM] LM Studio NICHT erreichbar ({self.api_url}): {e}")
            return False
