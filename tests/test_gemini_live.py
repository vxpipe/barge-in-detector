"""Live smoke test against Gemini. Excluded by default; run with
`uv run pytest -m integration tests/test_gemini_live.py`. Needs GEMINI_API_KEY."""

import os

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import GeminiClient
from barge_in_detector.generation.responses import parse_decision, parse_generation

pytestmark = pytest.mark.integration

SPEC = {
    "language": "en",
    "regional_variant": "en-IN",
    "script_form": "native",
    "code_switching": "within_sentence",
    "code_switch_set": "hi",
    "mixed_in_script": "default",
    "domain": "banking",
    "scenario": "banking_card_block",
    "speaker_pattern": "returning_speaker",
    "agent_activity": "explaining",
    "cut_position": "middle",
    "caller_intent": "agreement_token",
    "development": "stable",
    "repetition": "repeated",
    "length_bucket": "w2_3",
    "register": "casual",
    "caller_mood": "neutral",
}


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")), reason="no API key"
)
def test_generate_two_samples_and_judge_them_with_decisions():
    cat = load_catalogue()
    config = GenerationConfig.from_file()
    client = GeminiClient(explicit_cache=config.explicit_cache, cache_ttl_seconds=600)
    try:
        result = client.generate(
            model=config.generator_model,
            system_prompt=prompts.render_generator_system(cat),
            user_prompt=prompts.render_generation_request(cat, [SPEC], count=2),
            schema=prompts.GENERATOR_SCHEMA,
        )
        parsed = parse_generation(cat, result.text, specs=[SPEC], count=2)
        assert parsed.fatal is None
        assert parsed.samples, parsed.problems
        for i, s in enumerate(parsed.samples):
            item = prompts.JudgeItem(
                spec=SPEC,
                context_speaker=s.data["context_turn"]["speaker"],
                context_text=s.data["context_turn"]["text"],
                speaker=s.data["speaker"],
                speaker_spoken_text=s.data["speaker_spoken_text"],
                listener=s.data["listener"],
                listener_text=s.data["listener_text"],
            )
            judge_input, questions = prompts.render_judge_decision(cat, item)
            decided = client.decide(
                model=config.judge_model, input=judge_input, questions=questions
            )
            verdict = parse_decision(decided.response)
            print(f"\n{judge_input}\nsample {i} -> {verdict.label} ({verdict.notes})")
            assert verdict.label in ("backchannel", "barge_in", "ambiguous")
    finally:
        client.close()
