"""The storage module: the only writer of the working store.

It allows only the updates listed in docs/data-format.md ("What may change and
what may not"); there is deliberately no general update method. It also owns
the deterministic Parquet export and the import that rebuilds the database.
"""

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from barge_in_detector import ids
from barge_in_detector.catalogue import Catalogue
from barge_in_detector.generation.plan import ExistingTarget, Plan
from barge_in_detector.storage.schema import (
    PARTITION_DIRS,
    SCHEMA_VERSION,
    SPEC_FACTORS,
    TABLES,
    TURNS,
    Table,
    ddl,
)

ROWS_PER_PART = 200_000
ROW_GROUP_SIZE = 50_000


class RefusedUpdate(RuntimeError):
    """An update the data format does not allow."""


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


class Store:
    def __init__(self, con: duckdb.DuckDBPyConnection, path: Path | None):
        self.con = con
        self.path = path

    @classmethod
    def open(cls, path: Path) -> Store:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(path))
        store = cls(con, path)
        store._create()
        return store

    def _create(self) -> None:
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        for table in TABLES.values():
            self.con.execute(ddl(table))
        row = self.con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self.con.execute("INSERT INTO meta VALUES ('schema_version', ?)", [SCHEMA_VERSION])
        elif row[0] != SCHEMA_VERSION:
            raise RuntimeError(f"database schema {row[0]}, code expects {SCHEMA_VERSION}")

    def cursor(self) -> Store:
        """A store on its own cursor, for use from another thread."""
        return Store(self.con.cursor(), self.path)

    def close(self) -> None:
        self.con.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.con.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")

    # ---------------------------------------------------------------- reads

    def _dicts(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, list(params))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def count(self, table: str) -> int:
        return self.con.execute(f"SELECT count(*) FROM {TABLES[table].name}").fetchone()[0]

    def targets(
        self, *, status: str = "active", languages: Sequence[str] | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM targets WHERE status = ?"
        params: list[Any] = [status]
        if languages:
            sql += f" AND language IN ({', '.join('?' for _ in languages)})"
            params += list(languages)
        return self._dicts(sql + " ORDER BY priority, quota_group, combination_id", params)

    def existing_targets(self) -> list[ExistingTarget]:
        return [
            ExistingTarget(
                combination_id=r["combination_id"],
                split=r["split"],
                split_group_id=r["split_group_id"],
                quota_group=r["quota_group"],
                required_count=r["required_count"],
            )
            for r in self._dicts("SELECT * FROM targets")
        ]

    def request(self, request_id: uuid.UUID) -> dict[str, Any]:
        rows = self._dicts("SELECT * FROM requests WHERE request_id = ?", [request_id])
        if not rows:
            raise KeyError(request_id)
        row = rows[0]
        if row["response"] is not None:
            row["response"] = json.loads(row["response"])
        return row

    def requests(self, *, kind: str, status: str) -> list[dict[str, Any]]:
        rows = self._dicts(
            "SELECT * FROM requests WHERE kind = ? AND status = ? ORDER BY request_id",
            [kind, status],
        )
        for row in rows:
            if row["response"] is not None:
                row["response"] = json.loads(row["response"])
        return rows

    def judge_request_for(self, parent_id: uuid.UUID) -> dict[str, Any] | None:
        rows = self._dicts(
            "SELECT request_id FROM requests WHERE kind = 'judge' AND parent_request_id = ? "
            "AND status IN ('received', 'processed') ORDER BY request_id DESC LIMIT 1",
            [parent_id],
        )
        return self.request(rows[0]["request_id"]) if rows else None

    def conversation(self, conversation_id: uuid.UUID) -> dict[str, Any]:
        return self._dicts(
            "SELECT * FROM conversations WHERE conversation_id = ?", [conversation_id]
        )[0]

    def conversations(self, *, status: str = "active") -> list[dict[str, Any]]:
        return self._dicts(
            "SELECT * FROM conversations WHERE status = ? ORDER BY conversation_id", [status]
        )

    def accepted_counts(self) -> dict[uuid.UUID, int]:
        rows = self.con.execute(
            "SELECT combination_id, count(*) FROM conversations WHERE status = 'active' GROUP BY 1"
        ).fetchall()
        return dict(rows)

    def attempts(self) -> tuple[dict[uuid.UUID, int], dict[str, int]]:
        """Samples requested so far (API errors excluded), per criterion and per group."""
        by_criterion = dict(
            self.con.execute(
                "SELECT combination_id, sum(requested_count) FROM requests "
                "WHERE kind = 'generate' AND status != 'api_error' "
                "AND combination_id IS NOT NULL GROUP BY 1"
            ).fetchall()
        )
        by_group = dict(
            self.con.execute(
                "SELECT contrastive_group_id, sum(requested_count) FROM requests "
                "WHERE kind = 'generate' AND status != 'api_error' "
                "AND contrastive_group_id IS NOT NULL GROUP BY 1"
            ).fetchall()
        )
        return by_criterion, by_group

    def in_flight(self) -> tuple[dict[uuid.UUID, int], dict[str, int]]:
        """Samples in generate requests not yet processed, per criterion and per group."""
        rows = self.con.execute(
            "SELECT combination_id, contrastive_group_id, sum(requested_count) FROM requests "
            "WHERE kind = 'generate' AND status IN ('submitted', 'received') GROUP BY 1, 2"
        ).fetchall()
        by_criterion = {c: n for c, g, n in rows if c is not None}
        by_group = {g: n for c, g, n in rows if g is not None}
        return by_criterion, by_group

    def text_hashes(self) -> set[str]:
        rows = self.con.execute(
            "SELECT text_hash FROM conversations WHERE status = 'active'"
        ).fetchall()
        return {r[0] for r in rows}

    # ---------------------------------------------------------------- plan

    def apply_plan(self, cat: Catalogue, plan: Plan) -> uuid.UUID | None:
        """Record a plan revision, insert new criteria and raise required counts.

        Returns None, and records nothing, when the plan changes no target.
        """
        revision_id = ids.new_id()
        config = plan.config
        record = config.as_record()
        existing = {r["combination_id"]: r for r in self._dicts("SELECT * FROM targets")}
        if all(
            t.combination_id in existing
            and existing[t.combination_id]["required_count"] == t.required_count
            for t in plan.targets
        ):
            return None
        with self.transaction():
            self.con.execute(
                "INSERT INTO plan_revisions VALUES (?, ?, ?, ?, ?::JSON, ?::JSON, ?::JSON, ?::JSON)",
                [
                    revision_id,
                    config.name,
                    now(),
                    config.seed,
                    json.dumps(record["quotas"], sort_keys=True),
                    json.dumps(record["split_ratios"], sort_keys=True),
                    json.dumps(record["heldout"], sort_keys=True),
                    json.dumps(cat.file_versions, sort_keys=True),
                ],
            )
            for t in plan.targets:
                old = existing.get(t.combination_id)
                if old is None:
                    row = {
                        "combination_id": t.combination_id,
                        "factor_catalogue_version": cat.version,
                        **{f: t.spec[f] for f in SPEC_FACTORS},
                        "agent_state": t.agent_state,
                        "label": t.intended_label,
                        "contrastive_group_id": t.contrastive_group_id,
                        "split_group_id": t.split_group_id,
                        "split": t.split,
                        "contrastive_role": t.contrastive_role,
                        "required_count": t.required_count,
                        "quota_group": t.quota_group,
                        "priority": t.priority,
                        "added_in_revision": revision_id,
                        "count_set_in_revision": revision_id,
                        "status": "active",
                        "retired_reason": None,
                        "retired_at": None,
                    }
                    self._insert("targets", row)
                elif old["required_count"] != t.required_count:
                    self.con.execute(
                        "UPDATE targets SET required_count = ?, count_set_in_revision = ? "
                        "WHERE combination_id = ?",
                        [t.required_count, revision_id, t.combination_id],
                    )
        return revision_id

    def retire_target(self, combination_id: uuid.UUID, reason: str) -> None:
        self._retire("targets", "combination_id", combination_id, reason)

    # ---------------------------------------------------------------- runs

    def start_run(
        self,
        *,
        name: str,
        revision_id: uuid.UUID | None,
        settings: Mapping[str, Any],
        generator_model_id: str,
        judge_model_id: str,
    ) -> uuid.UUID:
        run_id = ids.new_id()
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?::JSON, 0, 0, 0, 0.0)",
            [
                run_id,
                name,
                revision_id,
                now(),
                generator_model_id,
                judge_model_id,
                json.dumps(settings, sort_keys=True),
            ],
        )
        return run_id

    def finish_run(self, run_id: uuid.UUID) -> None:
        self.con.execute(
            """
            UPDATE runs SET
              finished_at = ?,
              requests_made = (SELECT count(*) FROM requests WHERE run_id = ?),
              conversations_accepted = (SELECT count(*) FROM conversations WHERE run_id = ?),
              rejections = (SELECT count(*) FROM rejections WHERE run_id = ?),
              cost_usd = (SELECT coalesce(sum(cost_usd), 0) FROM requests WHERE run_id = ?)
            WHERE run_id = ?
            """,
            [now(), run_id, run_id, run_id, run_id, run_id],
        )

    # ---------------------------------------------------------------- requests

    def insert_request(
        self,
        *,
        run_id: uuid.UUID,
        kind: str,
        combination_id: uuid.UUID | None,
        contrastive_group_id: str | None,
        language: str,
        requested_count: int,
        model_id: str,
        prompt_template: str,
        prompt_hash: str,
        parent_request_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Insert a request before its call is made, so a crash never loses it."""
        request_id = ids.new_id()
        self._insert(
            "requests",
            {
                "request_id": request_id,
                "kind": kind,
                "parent_request_id": parent_request_id,
                "run_id": run_id,
                "combination_id": combination_id,
                "contrastive_group_id": contrastive_group_id,
                "language": language,
                "requested_count": requested_count,
                "batch_job_id": None,
                "model_id": model_id,
                "prompt_template": prompt_template,
                "prompt_hash": prompt_hash,
                "status": "submitted",
                "error": None,
                "response": None,
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "created_at": now(),
                "completed_at": None,
            },
        )
        return request_id

    def _status(self, request_id: uuid.UUID) -> str:
        row = self.con.execute(
            "SELECT status FROM requests WHERE request_id = ?", [request_id]
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return row[0]

    def record_response(
        self,
        request_id: uuid.UUID,
        *,
        response: Mapping[str, Any],
        usage: Mapping[str, int | None],
        cost_usd: float | None,
    ) -> None:
        """Store the raw response once; it never changes afterwards."""
        if self._status(request_id) != "submitted":
            raise RefusedUpdate(f"request {request_id} already has a response")
        self.con.execute(
            "UPDATE requests SET status = 'received', response = ?::JSON, input_tokens = ?, "
            "cached_input_tokens = ?, output_tokens = ?, cost_usd = ?, completed_at = ? "
            "WHERE request_id = ?",
            [
                json.dumps(response, sort_keys=True, ensure_ascii=False),
                usage.get("input_tokens"),
                usage.get("cached_input_tokens"),
                usage.get("output_tokens"),
                cost_usd,
                now(),
                request_id,
            ],
        )

    def fail_request(self, request_id: uuid.UUID, *, status: str, error: str) -> None:
        if status not in ("api_error", "schema_invalid"):
            raise ValueError(status)
        current = self._status(request_id)
        allowed = {"api_error": {"submitted"}, "schema_invalid": {"submitted", "received"}}
        if current not in allowed[status]:
            raise RefusedUpdate(f"request {request_id} is {current}, cannot become {status}")
        self.con.execute(
            "UPDATE requests SET status = ?, error = ?, completed_at = coalesce(completed_at, ?) "
            "WHERE request_id = ?",
            [status, error[:2000], now(), request_id],
        )

    def mark_processed(self, request_id: uuid.UUID) -> None:
        if self._status(request_id) != "received":
            raise RefusedUpdate(f"request {request_id} is not received")
        self.con.execute(
            "UPDATE requests SET status = 'processed' WHERE request_id = ?", [request_id]
        )

    # ---------------------------------------------------------------- conversations

    def insert_conversation(self, row: Mapping[str, Any]) -> uuid.UUID:
        target = row["target"]
        conversation_id = ids.new_id()
        values = {
            "conversation_id": conversation_id,
            **{c: target[c] for c, _ in TABLES["targets"].columns if c in _FACTOR_NAMES},
            "label": row["gemini_label"],
            "intended_label": target["label"],
            **{k: v for k, v in row.items() if k != "target"},
            "status": "active",
            "retired_reason": None,
            "retired_at": None,
            "created_at": now(),
        }
        self._insert("conversations", values)
        return conversation_id

    def retire_conversation(self, conversation_id: uuid.UUID, reason: str) -> None:
        self._retire("conversations", "conversation_id", conversation_id, reason)

    def insert_rejection(self, row: Mapping[str, Any]) -> uuid.UUID:
        rejection_id = ids.new_id()
        self._insert(
            "rejections",
            {"rejection_id": rejection_id, "judge_request_id": None, **row, "created_at": now()},
        )
        return rejection_id

    # ---------------------------------------------------------------- examples

    def replace_examples(self, builder_version: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Examples are derived; a rebuild replaces one builder version's rows."""
        with self.transaction():
            self.con.execute("DELETE FROM examples WHERE builder_version = ?", [builder_version])
            for row in rows:
                self._insert("examples", row)

    # ---------------------------------------------------------------- helpers

    def _insert(self, table_name: str, row: Mapping[str, Any]) -> None:
        table = TABLES[table_name]
        names = table.column_names
        unknown = set(row) - set(names)
        if unknown:
            raise ValueError(f"unknown columns for {table_name}: {sorted(unknown)}")
        placeholders = ", ".join(_placeholder(table.type_of(c)) for c in names)
        params = []
        for c in names:
            value = row.get(c)
            if table.type_of(c) == "JSON" and value is not None and not isinstance(value, str):
                value = json.dumps(value, sort_keys=True, ensure_ascii=False)
            params.append(value)
        self.con.execute(
            f"INSERT INTO {table_name} ({', '.join(names)}) VALUES ({placeholders})", params
        )

    def _retire(self, table: str, key: str, value: uuid.UUID, reason: str) -> None:
        row = self.con.execute(f"SELECT status FROM {table} WHERE {key} = ?", [value]).fetchone()
        if row is None:
            raise KeyError(value)
        if row[0] != "active":
            raise RefusedUpdate(f"{table} {value} is already {row[0]}")
        self.con.execute(
            f"UPDATE {table} SET status = 'retired', retired_reason = ?, retired_at = ? "
            f"WHERE {key} = ?",
            [reason, now(), value],
        )

    # ---------------------------------------------------------------- export / import

    def export(self, out_dir: Path) -> list[Path]:
        """Write every table to Parquet deterministically; return changed files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.con.execute("SET threads = 1")
        try:
            with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
                staged = Path(tmp)
                for table in TABLES.values():
                    self._export_table(table, staged)
                return _sync(staged, out_dir, managed=_managed_paths())
        finally:
            self.con.execute("RESET threads")

    def _export_table(self, table: Table, staged: Path) -> None:
        select = ", ".join(_export_expr(c, t) for c, t in table.columns)
        if not table.partitions:
            if self.count(table.name):
                self._copy(table, select, "TRUE", 0, staged / f"{table.name}.parquet")
            return
        cols = ", ".join(table.partitions)
        for values in self.con.execute(
            f"SELECT DISTINCT {cols} FROM {table.name} ORDER BY {cols}"
        ).fetchall():
            where = " AND ".join(
                f"{c} IS NULL" if v is None else f"CAST({c} AS VARCHAR) = {_literal(v)}"
                for c, v in zip(table.partitions, values, strict=True)
            )
            rows = self.con.execute(f"SELECT count(*) FROM {table.name} WHERE {where}").fetchone()[
                0
            ]
            directory = staged / table.name
            for c, v in zip(table.partitions, values, strict=True):
                directory /= f"{PARTITION_DIRS.get(c, c)}={v}"
            for part in range((rows + ROWS_PER_PART - 1) // ROWS_PER_PART):
                self._copy(table, select, where, part, directory / f"part-{part}.parquet")

    def _copy(self, table: Table, select: str, where: str, part: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(
            f"COPY (SELECT {select} FROM {table.name} WHERE {where} "
            f"ORDER BY {table.primary_key} LIMIT {ROWS_PER_PART} OFFSET {part * ROWS_PER_PART}) "
            f"TO {_literal(path)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP_SIZE})"
        )

    def import_exports(self, export_dir: Path) -> None:
        """Rebuild an empty database from Parquet exports."""
        export_dir = Path(export_dir)
        if any(self.count(t) for t in TABLES):
            raise RefusedUpdate("import needs an empty database")
        with self.transaction():
            for table in TABLES.values():
                pattern = (
                    export_dir / table.name / "**" / "*.parquet"
                    if table.partitions
                    else export_dir / f"{table.name}.parquet"
                )
                if not list(export_dir.glob(str(pattern.relative_to(export_dir)))):
                    continue
                select = ", ".join(f"CAST({c} AS {t}) AS {c}" for c, t in table.columns)
                self.con.execute(
                    f"INSERT INTO {table.name} SELECT {select} FROM "
                    f"read_parquet({_literal(pattern)}, hive_partitioning = false)"
                )


_FACTOR_NAMES = {c for c, _ in TABLES["conversations"].columns} & {
    c for c, _ in TABLES["targets"].columns
} - {"status", "retired_reason", "retired_at"}


def _placeholder(kind: str) -> str:
    if kind == "JSON":
        return "?::JSON"
    if kind == TURNS:
        return f"?::{TURNS}"
    return "?"


def _export_expr(column: str, kind: str) -> str:
    if kind in ("UUID", "JSON"):
        return f"CAST({column} AS VARCHAR) AS {column}"
    return column


def _managed_paths() -> list[str]:
    return [t.name if t.partitions else f"{t.name}.parquet" for t in TABLES.values()]


def _sync(staged: Path, out_dir: Path, *, managed: Sequence[str]) -> list[Path]:
    """Make the managed paths of out_dir equal staged, touching only changed files."""
    changed: list[Path] = []
    new_files = {p.relative_to(staged) for p in staged.rglob("*.parquet")}
    for name in managed:
        root = out_dir / name
        existing = (
            {p.relative_to(out_dir) for p in root.rglob("*.parquet")}
            if root.is_dir()
            else ({Path(name)} if root.is_file() else set())
        )
        for rel in existing - new_files:
            (out_dir / rel).unlink()
            changed.append(out_dir / rel)
    for rel in sorted(new_files):
        target = out_dir / rel
        source = staged / rel
        if target.is_file() and target.read_bytes() == source.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        changed.append(target)
    for name in managed:
        root = out_dir / name
        if root.is_dir():
            for directory in sorted(root.rglob("*"), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
    return changed
