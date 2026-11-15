"""Live smoke test against Gemini. Excluded by default; run with
`uv run pytest -m integration tests/test_gemini_live.py`. Needs GEMINI_API_KEY."""

import os

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import GeminiClient
from barge_in_detector.generation.responses import parse_utterances
from barge_in_detector.generation.utterance_criteria import judge_questions, parse_verdict
from barge_in_detector.generation.utterances import load_utterance_track

pytestmark = pytest.mark.integration

SPEC = {
    "language": "en",
    "regional_variant": "en-IN",
    "script_form": "native",
    "code_switching": "mixed",
    "code_switch_set": "hi",
    "mixed_in_script": "default",
    "caller_intent": "agreement_token",
    "development": "stable",
}


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")), reason="no API key"
)
def test_generate_two_utterances_and_judge_them():
    cat = load_catalogue()
    config = GenerationConfig.from_file()
    client = GeminiClient(explicit_cache=config.explicit_cache, cache_ttl_seconds=600)
    try:
        result = client.generate(
            model=config.generator_model,
            system_prompt=prompts.render_utterance_system(cat, load_utterance_track(cat)),
            user_prompt=prompts.render_utterance_request(cat, SPEC, count=2),
            schema=prompts.UTTERANCE_SCHEMA,
        )
        parsed = parse_utterances(cat, result.text, spec=SPEC, count=2)
        assert parsed.fatal is None
        assert parsed.samples, parsed.problems
        for s in parsed.samples:
            judge_input, questions = judge_questions(cat, SPEC, s.data["text"])
            decided = client.decide(
                model=config.judge_model, input=judge_input, questions=questions
            )
            verdict = parse_verdict(decided.response, list(questions))
            print(f"\n{s.data['text']} -> failed: {verdict.failed}")
    finally:
        client.close()
