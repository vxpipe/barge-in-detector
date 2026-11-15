"""Generation run settings (generation/config.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "generation" / "config.yaml"


@dataclass(frozen=True)
class GenerationConfig:
    generator_model: str = "gemini-3.5-flash-lite"
    judge_model: str = "gemini-decision-1"
    batch_size: int = 10
    max_attempts_per_sample: int = 3
    concurrency: int = 4
    temperature: float | None = None
    thinking_level: str | None = None
    explicit_cache: bool = True
    cache_ttl_seconds: int = 3600
    # model -> {input, cached_input, output} in USD per million tokens.
    prices_usd_per_million_tokens: dict[str, dict[str, float | None]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_PATH) -> GenerationConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def _prices(self, model: str) -> dict[str, float | None]:
        return self.prices_usd_per_million_tokens.get(model) or {}

    @property
    def has_prices(self) -> bool:
        """Prices are known for the generator and the judge model."""
        return all(
            all(self._prices(m).get(k) is not None for k in ("input", "cached_input", "output"))
            for m in (self.generator_model, self.judge_model)
        )

    def cost(self, usage: dict[str, int | None], model: str) -> float | None:
        p = self._prices(model)
        if any(p.get(k) is None for k in ("input", "cached_input", "output")):
            return None
        cached = usage.get("cached_input_tokens") or 0
        fresh = max((usage.get("input_tokens") or 0) - cached, 0)
        output = usage.get("output_tokens") or 0
        return (fresh * p["input"] + cached * p["cached_input"] + output * p["output"]) / 1e6
