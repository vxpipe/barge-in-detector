"""Table definitions for the working store (docs/data-format.md).

Two tables (user decision, 2026-11-15):

- `utterances`: every generated sample, accepted or rejected, with its
  criterion's factor values, the raw sample, the judge's raw answers and the
  verdict. It is the only table exported (data/export/utterances.csv).
- `requests`: the local request log (raw responses, tokens, cost, status) for
  resume, the attempt budget and costs. Not exported.

The criteria are not stored: they are built from the plan file in git.
Changing a column needs a SCHEMA_VERSION bump recorded in docs/data-format.md.
"""

from dataclasses import dataclass

# 6: two tables, utterances (every sample) and requests (local); the
# conversation track's tables, runs and plan revisions are gone.
SCHEMA_VERSION = "6"

# Factors of an utterance criterion (generation/factors/utterances.yaml).
UTTERANCE_FACTORS = (
    "language",
    "regional_variant",
    "script_form",
    "code_switching",
    "code_switch_set",
    "mixed_in_script",
    "caller_intent",
    "development",
    "repetition",
    "length_bucket",
)

STATUSES = ("accepted", "rejected", "retired")


@dataclass(frozen=True)
class Table:
    name: str
    primary_key: str
    columns: tuple[tuple[str, str], ...]

    @property
    def column_names(self) -> list[str]:
        return [c for c, _ in self.columns]

    def type_of(self, column: str) -> str:
        return dict(self.columns)[column]


UTTERANCES = Table(
    "utterances",
    "utterance_id",
    (
        ("utterance_id", "UUID"),
        ("combination_id", "UUID"),
        *((f, "VARCHAR") for f in UTTERANCE_FACTORS),
        ("locales", "VARCHAR[]"),
        ("split", "VARCHAR"),
        ("intended_label", "VARCHAR"),
        ("status", "VARCHAR"),
        ("reason", "VARCHAR"),
        ("text", "VARCHAR"),
        ("text_clean", "VARCHAR"),
        ("cleaner_version", "VARCHAR"),
        ("label", "VARCHAR"),
        ("barge_in_from_word", "INTEGER"),
        ("raw_sample", "JSON"),
        ("judge_answers", "JSON"),
        ("text_hash", "VARCHAR"),
        ("request_id", "UUID"),
        ("judge_request_id", "UUID"),
        ("sample_index", "INTEGER"),
        ("model_id", "VARCHAR"),
        ("prompt_template", "VARCHAR"),
        ("judge_model_id", "VARCHAR"),
        ("judge_template", "VARCHAR"),
        ("created_at", "TIMESTAMP"),
    ),
)

REQUESTS = Table(
    "requests",
    "request_id",
    (
        ("request_id", "UUID"),
        ("kind", "VARCHAR"),
        ("parent_request_id", "UUID"),
        ("combination_id", "UUID"),
        ("language", "VARCHAR"),
        ("requested_count", "INTEGER"),
        ("model_id", "VARCHAR"),
        ("prompt_template", "VARCHAR"),
        ("prompt_hash", "VARCHAR"),
        ("status", "VARCHAR"),
        ("error", "VARCHAR"),
        ("response", "JSON"),
        ("input_tokens", "INTEGER"),
        ("cached_input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        ("cost_usd", "DOUBLE"),
        ("created_at", "TIMESTAMP"),
        ("completed_at", "TIMESTAMP"),
    ),
)

TABLES: dict[str, Table] = {t.name: t for t in (UTTERANCES, REQUESTS)}


def ddl(table: Table) -> str:
    cols = ",\n  ".join(
        f"{name} {kind}{' PRIMARY KEY' if name == table.primary_key else ''}"
        for name, kind in table.columns
    )
    return f"CREATE TABLE IF NOT EXISTS {table.name} (\n  {cols}\n)"
