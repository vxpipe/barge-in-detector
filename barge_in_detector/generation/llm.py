"""LLM client boundary. The runner depends on the LlmClient protocol only, so
tests use a fake and live runs use GeminiClient.

- generate(): the generator, through the Gemini generateContent API.
- decide(): the judge, through the Gemini Decisions API (questions in,
  calibrated probabilities out, no generated text).

Never log or store API keys or request headers: only the curated response
fields below are kept.
"""

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_RETRYABLE = {429, 500, 502, 503, 504}
_BLOCK_CODES = ("prohibited_content", "content_blocked", "image_safety", "blocklist", "safety")


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


@dataclass(frozen=True)
class DecisionResult:
    """`response` holds `answers` (question key -> answer fields) or `blocked`."""

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

    def decide(self, *, model: str, input: str, questions: Mapping[str, Any]) -> DecisionResult: ...

    def close(self) -> None: ...


def _status(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


class GeminiClient:
    """google-genai client.

    The generator's system prompt goes into an explicit context cache per
    (model, prompt hash). If a model refuses explicit caching, caching is
    switched off for that model for the rest of the run and the prompt is sent
    as system_instruction, so implicit prefix caching can still apply.
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
        # v1beta: context caching (cachedContents) and the Decisions API exist
        # only there; google-genai 3.x defaults to v1, where creating a cache
        # returns 404.
        self._client = genai.Client(http_options={"api_version": "v1beta"})
        self._decisions_client = self._client
        self._temperature = temperature
        self._explicit_cache = explicit_cache
        self._ttl = cache_ttl_seconds
        self._max_retries = max_retries
        self._caches: dict[tuple[str, str], str | None] = {}
        self._no_cache_models: set[str] = set()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- retries

    def _call(
        self, fn: Callable[[], Any], *, on_client_error: Callable[[Exception], bool] | None = None
    ) -> Any:
        """Call fn, retrying rate limits, server and network errors with backoff.

        A client error (4xx other than 429) is retried only when
        on_client_error handles it and returns True.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                status = _status(exc)
                if status is not None and status < 500 and status != 429:
                    if on_client_error is not None and on_client_error(exc):
                        continue
                    message = getattr(exc, "message", None) or str(exc)
                    raise LlmError(redact(f"{status}: {message}")[:1000]) from None
                if attempt < self._max_retries:
                    attempt += 1
                    time.sleep(min(60, 2**attempt))
                    continue
                raise LlmError(redact(f"{status or type(exc).__name__}: {exc}")[:1000]) from None

    # ---------------------------------------------------------------- generator

    def _cache_key(self, model: str, system_prompt: str) -> tuple[str, str]:
        return (model, hashlib.sha256(system_prompt.encode()).hexdigest())

    def _cache_name(self, model: str, system_prompt: str) -> str | None:
        from google.genai import types

        key = self._cache_key(model, system_prompt)
        with self._lock:
            if model in self._no_cache_models:
                return None
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
                self._no_cache_models.add(model)
                log.warning(
                    "explicit context caching unavailable for %s (%s); sending the system "
                    "prompt with each call, implicit caching may still apply",
                    model,
                    _status(exc) or type(exc).__name__,
                )
            self._caches[key] = name
            return name

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        thinking_level: str | None = None,
    ) -> LlmResult:
        from google.genai import types

        state: dict[str, Any] = {"cache": None, "refreshed": False}

        def call() -> Any:
            state["cache"] = (
                self._cache_name(model, system_prompt) if self._explicit_cache else None
            )
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=dict(schema),
                temperature=self._temperature,
                cached_content=state["cache"],
                system_instruction=None if state["cache"] else system_prompt,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
                ),
            )
            return self._client.models.generate_content(
                model=model, contents=user_prompt, config=config
            )

        def cache_expired(exc: Exception) -> bool:
            # A cache that expired or was deleted mid-run: recreate it once.
            if state["cache"] and not state["refreshed"] and _status(exc) in (400, 403, 404):
                state["refreshed"] = True
                with self._lock:
                    self._caches.pop(self._cache_key(model, system_prompt), None)
                return True
            return False

        response = self._call(call, on_client_error=cache_expired)
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
            "explicit_cache": bool(state["cache"]),
        }
        if not text:
            raise LlmError(f"empty response (finish_reason={finish})")
        return LlmResult(text=text, response=curated, usage=usage)

    # ---------------------------------------------------------------- judge

    def decide(self, *, model: str, input: str, questions: Mapping[str, Any]) -> DecisionResult:
        """One Decisions API request. A blocked input is a result, not an error."""
        try:
            result = self._call(
                lambda: self._decisions_client.decisions.create(
                    model=model, input=input, questions=dict(questions)
                )
            )
        except LlmError as exc:
            message = str(exc).lower()
            code = next((c for c in _BLOCK_CODES if c in message), None)
            if message.startswith("400") and code:
                return DecisionResult(response={"blocked": code}, usage={})
            raise
        feedback = getattr(result, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            return DecisionResult(response={"blocked": str(feedback.block_reason)}, usage={})
        usage_meta = getattr(result, "usage", None)
        usage = {
            "input_tokens": getattr(usage_meta, "total_input_tokens", None),
            "cached_input_tokens": getattr(usage_meta, "total_cached_tokens", None),
            "output_tokens": getattr(usage_meta, "total_output_tokens", None) or 0,
        }
        answers = {
            key: answer.model_dump(mode="json", exclude_none=True)
            for key, answer in (result.answers or {}).items()
        }
        return DecisionResult(response={"answers": answers, "usage": usage}, usage=usage)

    def close(self) -> None:
        for name in {n for n in self._caches.values() if n}:
            try:
                self._client.caches.delete(name=name)
            except Exception as exc:  # noqa: BLE001 - best effort; the TTL expires it anyway
                log.warning("could not delete context cache: %s", redact(str(exc))[:200])
        self._caches.clear()
