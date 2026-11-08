"""Online-First with Local-Backup LLM Client for P.E.R.S.O.N.A. Engine.

Primary: Cloud LLM APIs (OpenAI-compatible, Google Gemini, Anthropic Claude).
Backup: Local Ollama inference server (`http://localhost:11434/api/chat`) on the Linux host,
followed by deterministic persona synthesis if both cloud and local inference are unreachable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMClient:
    """Client for persona language generation: Online-First, Local Ollama as Backup."""

    def __init__(
        self,
        online_provider: str = "openai_compatible",
        online_model: str = "gpt-4o",
        gemini_model: str = "gemini-2.5-pro",
        anthropic_model: str = "claude-sonnet-4-20250514",
        backup_provider: str = "ollama",
        ollama_url: str = "http://localhost:11434",
        backup_model: str = "llama3.1:8b",
        temperature: float = 0.6,
        timeout_seconds: float = 6.0,
    ) -> None:
        self.online_provider = online_provider
        self.online_model = online_model
        self.gemini_model = gemini_model
        self.anthropic_model = anthropic_model
        self.backup_provider = backup_provider
        self.ollama_url = os.environ.get("OLLAMA_HOST", ollama_url).rstrip("/")
        self.backup_model = backup_model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        persona_id: str,
        speaker_context: dict[str, Any] | None = None,
    ) -> str | None:
        """Query Online LLM APIs first; fall back to local Ollama server if offline/unreachable."""
        if os.environ.get("VANGUARD_OFFLINE_MODE") == "1":
            return None

        # 1. ONLINE PRIMARY: Try configured online provider first, then any other configured cloud keys
        online_order = [self.online_provider]
        for candidate in ("openai_compatible", "gemini", "anthropic"):
            if candidate not in online_order:
                online_order.append(candidate)

        for provider in online_order:
            if provider == "openai_compatible" and os.environ.get("OPENAI_API_KEY"):
                res = self._call_openai(system_prompt, user_message)
                if res:
                    return res
            elif provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
                res = self._call_gemini(system_prompt, user_message)
                if res:
                    return res
            elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                res = self._call_anthropic(system_prompt, user_message)
                if res:
                    return res

        # 2. LOCAL BACKUP: Fall back to local Ollama server on the Linux host
        if self.backup_provider == "ollama":
            return self._call_ollama(system_prompt, user_message)

        return None

    def _call_openai(self, system_prompt: str, user_message: str) -> str | None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.online_model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip() or None
        except Exception:
            return None
        return None

    def _call_gemini(self, system_prompt: str, user_message: str) -> str | None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip() or None
        except Exception:
            return None
        return None

    def _call_anthropic(self, system_prompt: str, user_message: str) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.anthropic_model,
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "").strip() or None
        except Exception:
            return None
        return None

    def _call_ollama(self, system_prompt: str, user_message: str) -> str | None:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.backup_model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data.get("message", {}).get("content", "").strip()
                return msg if msg else None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
