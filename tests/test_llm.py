from barge_in_detector.generation import llm


def test_gemini_client_uses_v1beta_for_caching_and_decisions(monkeypatch):
    # Context caching (cachedContents) and the Decisions API exist only in
    # v1beta; google-genai 3.x defaults to v1, where cache creation returns 404.
    made = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm.GeminiClient()
    assert made and all(m.get("http_options", {}).get("api_version") == "v1beta" for m in made)


def test_redact_removes_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-123")
    assert llm.redact("bad key secret-123 here") == "bad key [redacted] here"


def test_bounded_generation_caps_output_and_disables_sdk_retries(monkeypatch):
    from types import SimpleNamespace

    from google import genai

    made, calls = [], []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=10,
                thoughts_token_count=3,
                cached_content_token_count=0,
            ),
            candidates=[],
            text='{"samples": []}',
            model_version="m",
            response_id="r",
        )

    def client(**kwargs):
        made.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    result = llm.GeminiClient(explicit_cache=False, max_retries=0).generate(
        model="m",
        system_prompt="s",
        user_prompt="u",
        schema={},
        max_output_tokens=256,
    )
    assert made[0]["http_options"]["retry_options"]["attempts"] == 1
    assert calls[0]["config"].max_output_tokens == 256
    assert result.usage["output_tokens"] == 13
