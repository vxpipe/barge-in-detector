"""LLM client boundary. The runner depends on the LlmClient protocol only, so
tests use a fake and live runs use GeminiClient.

Never log or store API keys or request headers: only the curated response
fields below are kept.
"""

import hashlib
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def redact(text: str) -> str:
    for var in _KEY_VARS:
        value = os.environ.get(var)
        if value:
            text = text.replace(value, "[redacted]")
    return text


@dataclass(frozen=True)
class LlmResult:
    text: str
    response: dict[str, Any]
    usage: dict[str, int | None]


class LlmError(RuntimeError):
    """A call failed; the message is safe to store."""


class LlmClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        thinking_level: str | None = None,
    ) -> LlmResult: ...

    def close(self) -> None: ...


_RETRYABLE = {429, 500, 502, 503, 504}


class GeminiClient:
    """google-genai client with a cached system prompt.

    The system prompt goes into an explicit context cache per (model, prompt
    hash). If the cache cannot be created, the prompt is sent as
    system_instruction on every call, so implicit prefix caching can apply.
    """

    def __init__(
        self,
        *,
        temperature: float | None = None,
        explicit_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        max_retries: int = 4,
    ):
        from google import genai

        if not any(os.environ.get(v) for v in _KEY_VARS):
            raise LlmError("set GEMINI_API_KEY or GOOGLE_API_KEY")
        self._client = genai.Client()
        self._temperature = temperature
        self._explicit_cache = explicit_cache
        self._ttl = cache_ttl_seconds
        self._max_retries = max_retries
        self._caches: dict[tuple[str, str], str | None] = {}
        self._lock = threading.Lock()

    def _cache_name(self, model: str, system_prompt: str) -> str | None:
        from google.genai import types

        key = (model, hashlib.sha256(system_prompt.encode()).hexdigest())
        with self._lock:
            if key in self._caches:
                return self._caches[key]
            name = None
            try:
                cache = self._client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_prompt,
                        ttl=f"{self._ttl}s",
                        display_name=f"barge-in-{key[1][:16]}",
                    ),
                )
                name = cache.name
                log.info("created context cache for %s (%s)", model, key[1][:12])
            except Exception as exc:  # noqa: BLE001 - any failure falls back to no cache
                log.warning(
                    "context cache unavailable for %s, sending the system prompt each call: %s",
                    model,
                    redact(str(exc))[:300],
                )
            self._caches[key] = name
            return name

    def _forget_cache(self, model: str, system_prompt: str) -> None:
        key = (model, hashlib.sha256(system_prompt.encode()).hexdigest())
        with self._lock:
            self._caches.pop(key, None)

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        thinking_level: str | None = None,
    ) -> LlmResult:
        from google.genai import errors, types

        attempt = 0
        cache_refreshed = False
        while True:
            cache = self._cache_name(model, system_prompt) if self._explicit_cache else None
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=dict(schema),
                temperature=self._temperature,
                cached_content=cache,
                system_instruction=None if cache else system_prompt,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
                ),
            )
            try:
                response = self._client.models.generate_content(
                    model=model, contents=user_prompt, config=config
                )
                break
            except errors.APIError as exc:
                if cache and not cache_refreshed and exc.code in (400, 403, 404):
                    # The cache may have expired or been deleted: recreate it once.
                    cache_refreshed = True
                    self._forget_cache(model, system_prompt)
                    continue
                if exc.code in _RETRYABLE and attempt < self._max_retries:
                    attempt += 1
                    time.sleep(min(60, 2**attempt))
                    continue
                raise LlmError(redact(f"{exc.code}: {exc.message}")[:1000]) from None
            except Exception as exc:  # noqa: BLE001 - network errors
                if attempt < self._max_retries:
                    attempt += 1
                    time.sleep(min(60, 2**attempt))
                    continue
                raise LlmError(redact(f"{type(exc).__name__}: {exc}")[:1000]) from None

        usage_meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(usage_meta, "prompt_token_count", None),
            "cached_input_tokens": getattr(usage_meta, "cached_content_token_count", None),
            "output_tokens": (getattr(usage_meta, "candidates_token_count", None) or 0)
            + (getattr(usage_meta, "thoughts_token_count", None) or 0),
        }
        candidate = response.candidates[0] if response.candidates else None
        finish = str(candidate.finish_reason) if candidate and candidate.finish_reason else None
        text = response.text or ""
        curated = {
            "text": text,
            "model_version": response.model_version,
            "response_id": response.response_id,
            "finish_reason": finish,
            "usage": usage,
            "explicit_cache": bool(cache),
        }
        if not text:
            raise LlmError(f"empty response (finish_reason={finish})")
        return LlmResult(text=text, response=curated, usage=usage)

    def close(self) -> None:
        for name in {n for n in self._caches.values() if n}:
            try:
                self._client.caches.delete(name=name)
            except Exception as exc:  # noqa: BLE001 - best effort; the TTL expires it anyway
                log.warning("could not delete context cache: %s", redact(str(exc))[:200])
        self._caches.clear()
