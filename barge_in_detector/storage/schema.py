"""Table definitions for the working store and the Parquet exports.

The columns are specified in docs/data-format.md. Changing a column needs a
SCHEMA_VERSION bump recorded there.
"""

from dataclasses import dataclass

# 1: initial. 2: `locales` (every locale in the transcript) on targets,
# conversations and examples. 3: the `mixed_in_script` factor; locales carry
# script subtags; examples.transcript_raw. 4: the utterance track's
# `utterance_targets` and `utterances` tables (docs/utterance-first-decision.md).
SCHEMA_VERSION = "4"

SPEC_FACTORS = (
    "language",
    "regional_variant",
    "script_form",
    "code_switching",
    "code_switch_set",
    "mixed_in_script",
    "domain",
    "scenario",
    "speaker_pattern",
    "agent_activity",
    "cut_position",
    "caller_intent",
    "development",
    "repetition",
    "length_bucket",
    "register",
    "caller_mood",
)

# Factors of an utterance-track criterion (generation/factors/utterances.yaml).
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

UTTERANCE_FACTOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("combination_id", "UUID"),
    ("factor_catalogue_version", "VARCHAR"),
    ("utterance_track_version", "VARCHAR"),
    *((f, "VARCHAR") for f in UTTERANCE_FACTORS),
    ("locales", "VARCHAR[]"),
    ("split", "VARCHAR"),
)

TURNS = "STRUCT(speaker VARCHAR, role VARCHAR, text VARCHAR)[]"

FACTOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("combination_id", "UUID"),
    ("factor_catalogue_version", "VARCHAR"),
    *((f, "VARCHAR") for f in SPEC_FACTORS),
    # The conversation's locale first, then each mixed-in language's, e.g.
    # [en-MY, zh-MY]. Derived from the spec; not part of combination_id.
    ("locales", "VARCHAR[]"),
    ("agent_state", "VARCHAR"),
    ("label", "VARCHAR"),
    ("contrastive_group_id", "VARCHAR"),
    ("split_group_id", "UUID"),
    ("split", "VARCHAR"),
)


@dataclass(frozen=True)
class Table:
    name: str
    primary_key: str
    columns: tuple[tuple[str, str], ...]
    partitions: tuple[str, ...] = ()

    @property
    def column_names(self) -> list[str]:
        return [c for c, _ in self.columns]

    def type_of(self, column: str) -> str:
        return dict(self.columns)[column]


# Directory names used for partition columns in exports.
PARTITION_DIRS = {"builder_version": "builder"}

TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        Table(
            "plan_revisions",
            "revision_id",
            (
                ("revision_id", "UUID"),
                ("name", "VARCHAR"),
                ("created_at", "TIMESTAMP"),
                ("seed", "BIGINT"),
                ("quotas", "JSON"),
                ("split_ratios", "JSON"),
                ("heldout", "JSON"),
                ("catalogue_versions", "JSON"),
            ),
        ),
        Table(
            "runs",
            "run_id",
            (
                ("run_id", "UUID"),
                ("name", "VARCHAR"),
                ("revision_id", "UUID"),
                ("started_at", "TIMESTAMP"),
                ("finished_at", "TIMESTAMP"),
                ("generator_model_id", "VARCHAR"),
                ("judge_model_id", "VARCHAR"),
                ("settings", "JSON"),
                ("requests_made", "INTEGER"),
                ("conversations_accepted", "INTEGER"),
                ("rejections", "INTEGER"),
                ("cost_usd", "DOUBLE"),
            ),
        ),
        Table(
            "targets",
            "combination_id",
            (
                *FACTOR_COLUMNS,
                ("contrastive_role", "INTEGER"),
                ("required_count", "INTEGER"),
                ("quota_group", "VARCHAR"),
                ("priority", "INTEGER"),
                ("added_in_revision", "UUID"),
                ("count_set_in_revision", "UUID"),
                ("status", "VARCHAR"),
                ("retired_reason", "VARCHAR"),
                ("retired_at", "TIMESTAMP"),
            ),
            partitions=("language",),
        ),
        Table(
            "requests",
            "request_id",
            (
                ("request_id", "UUID"),
                ("kind", "VARCHAR"),
                ("parent_request_id", "UUID"),
                ("run_id", "UUID"),
                ("combination_id", "UUID"),
                ("contrastive_group_id", "VARCHAR"),
                ("language", "VARCHAR"),
                ("requested_count", "INTEGER"),
                ("batch_job_id", "VARCHAR"),
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
            partitions=("language", "run_id"),
        ),
        Table(
            "conversations",
            "conversation_id",
            (
                ("conversation_id", "UUID"),
                *FACTOR_COLUMNS,
                ("request_id", "UUID"),
                ("judge_request_id", "UUID"),
                ("run_id", "UUID"),
                ("sample_index", "INTEGER"),
                ("turns", TURNS),
                ("agent_speaker", "VARCHAR"),
                ("agent_full_text", "VARCHAR"),
                ("agent_spoken_text", "VARCHAR"),
                ("candidate_speaker", "VARCHAR"),
                ("candidate_role", "VARCHAR"),
                ("candidate_text", "VARCHAR"),
                ("intended_label", "VARCHAR"),
                ("gemini_label", "VARCHAR"),
                ("gemini_rationale", "VARCHAR"),
                ("barge_in_from_word", "INTEGER"),
                ("judge_label", "VARCHAR"),
                ("judge_barge_in_from_word", "INTEGER"),
                ("judge_notes", "VARCHAR"),
                ("model_id", "VARCHAR"),
                ("prompt_template", "VARCHAR"),
                ("judge_model_id", "VARCHAR"),
                ("text_hash", "VARCHAR"),
                ("status", "VARCHAR"),
                ("retired_reason", "VARCHAR"),
                ("retired_at", "TIMESTAMP"),
                ("created_at", "TIMESTAMP"),
            ),
            partitions=("language", "run_id"),
        ),
        Table(
            "rejections",
            "rejection_id",
            (
                ("rejection_id", "UUID"),
                ("request_id", "UUID"),
                ("judge_request_id", "UUID"),
                ("combination_id", "UUID"),
                ("language", "VARCHAR"),
                ("run_id", "UUID"),
                ("sample_index", "INTEGER"),
                ("stage", "VARCHAR"),
                ("reason", "VARCHAR"),
                ("gemini_label", "VARCHAR"),
                ("judge_label", "VARCHAR"),
                ("created_at", "TIMESTAMP"),
            ),
            partitions=("language", "run_id"),
        ),
        Table(
            "examples",
            "example_id",
            (
                ("example_id", "UUID"),
                *FACTOR_COLUMNS,
                ("conversation_id", "UUID"),
                ("run_id", "UUID"),
                ("builder_version", "VARCHAR"),
                ("turn_shape", "TINYINT"),
                ("turns_original", TURNS),
                ("transcript_original", "VARCHAR"),
                ("transcript_clean", "VARCHAR"),
                # s<n> labels, text as written (NFC, whitespace only): the
                # alternative model input with casing and punctuation kept.
                ("transcript_raw", "VARCHAR"),
                ("speaker_labels", "VARCHAR[]"),
                ("candidate_original", "VARCHAR"),
                ("candidate_clean", "VARCHAR"),
                ("cleaner_version", "VARCHAR"),
                ("prefix_index", "SMALLINT"),
                ("is_final_prefix", "BOOLEAN"),
                ("label_id", "TINYINT"),
            ),
            partitions=("builder_version", "split", "language", "run_id"),
        ),
        Table(
            "utterance_targets",
            "combination_id",
            (
                *UTTERANCE_FACTOR_COLUMNS,
                ("label", "VARCHAR"),
                ("required_count", "INTEGER"),
                ("priority", "INTEGER"),
                ("added_in_revision", "UUID"),
                ("count_set_in_revision", "UUID"),
                ("status", "VARCHAR"),
                ("retired_reason", "VARCHAR"),
                ("retired_at", "TIMESTAMP"),
            ),
            partitions=("language",),
        ),
        Table(
            "utterances",
            "utterance_id",
            (
                ("utterance_id", "UUID"),
                *UTTERANCE_FACTOR_COLUMNS,
                ("label", "VARCHAR"),
                ("intended_label", "VARCHAR"),
                ("request_id", "UUID"),
                ("run_id", "UUID"),
                ("sample_index", "INTEGER"),
                ("text", "VARCHAR"),
                ("text_clean", "VARCHAR"),
                ("cleaner_version", "VARCHAR"),
                ("gemini_label", "VARCHAR"),
                ("gemini_rationale", "VARCHAR"),
                ("barge_in_from_word", "INTEGER"),
                ("judge_label", "VARCHAR"),
                ("judge_model_id", "VARCHAR"),
                ("model_id", "VARCHAR"),
                ("prompt_template", "VARCHAR"),
                ("text_hash", "VARCHAR"),
                ("status", "VARCHAR"),
                ("retired_reason", "VARCHAR"),
                ("retired_at", "TIMESTAMP"),
                ("created_at", "TIMESTAMP"),
            ),
            partitions=("language", "run_id"),
        ),
        Table(
            "annotations",
            "annotation_id",
            (
                ("annotation_id", "UUID"),
                ("example_id", "UUID"),
                ("source", "VARCHAR"),
                ("human_label", "VARCHAR"),
                ("ambiguous", "BOOLEAN"),
                ("annotator", "VARCHAR"),
                ("annotated_at", "TIMESTAMP"),
                ("annotation_notes", "VARCHAR"),
            ),
        ),
    )
}


def ddl(table: Table) -> str:
    cols = ",\n  ".join(
        f"{name} {kind}{' PRIMARY KEY' if name == table.primary_key else ''}"
        for name, kind in table.columns
    )
    return f"CREATE TABLE IF NOT EXISTS {table.name} (\n  {cols}\n)"
