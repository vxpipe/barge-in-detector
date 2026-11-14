"""Generation run settings (generation/config.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "generation" / "config.yaml"


@dataclass(frozen=True)
class GenerationConfig:
    generator_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.8-flash"
    batch_size: int = 10
    max_attempts_per_sample: int = 3
    concurrency: int = 4
    temperature: float | None = None
    thinking_level: str | None = None
    judge_thinking_level: str | None = None
    explicit_cache: bool = True
    cache_ttl_seconds: int = 3600
    barge_in_word_tolerance: int = 1
    prices_usd_per_million_tokens: dict[str, float | None] = field(
        default_factory=lambda: {"input": None, "cached_input": None, "output": None}
    )

    @classmethod
    def from_file(cls, path: Path = DEFAULT_PATH) -> GenerationConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def has_prices(self) -> bool:
        return all(v is not None for v in self.prices_usd_per_million_tokens.values())

    def cost(self, usage: dict[str, int | None]) -> float | None:
        if not self.has_prices:
            return None
        p = self.prices_usd_per_million_tokens
        cached = usage.get("cached_input_tokens") or 0
        fresh = max((usage.get("input_tokens") or 0) - cached, 0)
        output = usage.get("output_tokens") or 0
        return (fresh * p["input"] + cached * p["cached_input"] + output * p["output"]) / 1e6
