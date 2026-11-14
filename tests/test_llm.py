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
