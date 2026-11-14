"""Live smoke test against Gemini. Excluded by default; run with
`uv run pytest -m integration tests/test_gemini_live.py`. Needs GEMINI_API_KEY."""

import os

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import GeminiClient
from barge_in_detector.generation.responses import parse_generation, parse_judgement

pytestmark = pytest.mark.integration

SPEC = {
    "language": "en",
    "regional_variant": "en-IN",
    "script_form": "native",
    "code_switching": "within_sentence",
    "code_switch_set": "hi",
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
def test_generate_and_judge_two_samples():
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
        items = [
            prompts.JudgeItem(
                index=i,
                spec=SPEC,
                context_speaker=s.data["context_turn"]["speaker"],
                context_text=s.data["context_turn"]["text"],
                agent_speaker=s.data["agent_speaker"],
                agent_spoken_text=s.data["agent_spoken_text"],
                candidate_speaker=s.data["candidate_speaker"],
                candidate_text=s.data["candidate_text"],
            )
            for i, s in enumerate(parsed.samples)
        ]
        judged = client.generate(
            model=config.judge_model,
            system_prompt=prompts.render_judge_system(cat),
            user_prompt=prompts.render_judge_request(cat, items),
            schema=prompts.JUDGE_SCHEMA,
        )
        verdicts = parse_judgement(judged.text, count=len(items))
        assert set(verdicts) == set(range(len(items)))
    finally:
        client.close()
