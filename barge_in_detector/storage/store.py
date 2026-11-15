"""The storage module: the only writer of the working store (docs/data-format.md).

Generated text is never changed: a sample row is inserted once, and only its
status may change (accepted to retired). Requests are inserted before their
call and get their response once. The export writes `utterances` to one CSV,
deterministically; the request log stays local.
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
from barge_in_detector.cleaner import CLEANER_VERSION, clean_text
from barge_in_detector.storage.schema import (
    REQUESTS,
    SCHEMA_VERSION,
    TABLES,
    UTTERANCE_FACTORS,
    UTTERANCES,
    Table,
    ddl,
)

EXPORT_FILE = "utterances.csv"


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
        store = cls(duckdb.connect(str(path)), path)
        store._create()
        return store

    def _create(self) -> None:
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        row = self.con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None and row[0] != SCHEMA_VERSION:
            self.con.close()
            raise RuntimeError(
                f"database schema {row[0]}, code expects {SCHEMA_VERSION}: delete "
                f"{self.path} and rebuild it with `import` from data/export"
            )
        for table in TABLES.values():
            self.con.execute(ddl(table))
        if row is None:
            self.con.execute("INSERT INTO meta VALUES ('schema_version', ?)", [SCHEMA_VERSION])

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

    def _dicts(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, list(params))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def count(self, table: str) -> int:
        return self.con.execute(f"SELECT count(*) FROM {TABLES[table].name}").fetchone()[0]

    def _insert(self, table: Table, row: Mapping[str, Any]) -> None:
        unknown = set(row) - set(table.column_names)
        if unknown:
            raise ValueError(f"unknown columns for {table.name}: {sorted(unknown)}")
        placeholders, params = [], []
        for name, kind in table.columns:
            value = row.get(name)
            if kind == "JSON" and value is not None and not isinstance(value, str):
                value = json.dumps(value, sort_keys=True, ensure_ascii=False)
            placeholders.append("?::JSON" if kind == "JSON" else "?")
            params.append(value)
        self.con.execute(
            f"INSERT INTO {table.name} ({', '.join(table.column_names)}) "
            f"VALUES ({', '.join(placeholders)})",
            params,
        )

    # ---------------------------------------------------------------- utterances

    def insert_utterance(self, target: Mapping[str, Any], row: Mapping[str, Any]) -> uuid.UUID:
        """One generated sample, accepted or rejected; its text never changes."""
        if row["status"] not in ("accepted", "rejected"):
            raise ValueError(row["status"])
        utterance_id = ids.new_id()
        text = row.get("text")
        self._insert(
            UTTERANCES,
            {
                "utterance_id": utterance_id,
                "combination_id": target["combination_id"],
                **{f: target[f] for f in UTTERANCE_FACTORS},
                "locales": list(target["locales"]),
                "split": target["split"],
                "intended_label": target["label"],
                "text_clean": clean_text(text) if isinstance(text, str) else None,
                "cleaner_version": CLEANER_VERSION,
                "created_at": now(),
                **row,
            },
        )
        return utterance_id

    def utterances(self, *, status: str | None = "accepted") -> list[dict[str, Any]]:
        if status is None:
            return self._dicts("SELECT * FROM utterances ORDER BY utterance_id")
        return self._dicts(
            "SELECT * FROM utterances WHERE status = ? ORDER BY utterance_id", [status]
        )

    def accepted_counts(self) -> dict[uuid.UUID, int]:
        return dict(
            self.con.execute(
                "SELECT combination_id, count(*) FROM utterances "
                "WHERE status = 'accepted' GROUP BY 1"
            ).fetchall()
        )

    def accepted_text_hashes(self) -> set[str]:
        rows = self.con.execute(
            "SELECT text_hash FROM utterances WHERE status = 'accepted'"
        ).fetchall()
        return {r[0] for r in rows}

    def retire_utterance(self, utterance_id: uuid.UUID, reason: str) -> None:
        row = self.con.execute(
            "SELECT status FROM utterances WHERE utterance_id = ?", [utterance_id]
        ).fetchone()
        if row is None:
            raise KeyError(utterance_id)
        if row[0] != "accepted":
            raise RefusedUpdate(f"utterance {utterance_id} is {row[0]}")
        self.con.execute(
            "UPDATE utterances SET status = 'retired', reason = ? WHERE utterance_id = ?",
            [reason, utterance_id],
        )

    # ---------------------------------------------------------------- requests (local)

    def insert_request(
        self,
        *,
        kind: str,
        combination_id: uuid.UUID | None,
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
            REQUESTS,
            {
                "request_id": request_id,
                "kind": kind,
                "parent_request_id": parent_request_id,
                "combination_id": combination_id,
                "language": language,
                "requested_count": requested_count,
                "model_id": model_id,
                "prompt_template": prompt_template,
                "prompt_hash": prompt_hash,
                "status": "submitted",
                "created_at": now(),
            },
        )
        return request_id

    def _request_rows(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        rows = self._dicts(sql, params)
        for row in rows:
            if row["response"] is not None:
                row["response"] = json.loads(row["response"])
        return rows

    def request(self, request_id: uuid.UUID) -> dict[str, Any]:
        rows = self._request_rows("SELECT * FROM requests WHERE request_id = ?", [request_id])
        if not rows:
            raise KeyError(request_id)
        return rows[0]

    def requests(self, *, kind: str, status: str) -> list[dict[str, Any]]:
        return self._request_rows(
            "SELECT * FROM requests WHERE kind = ? AND status = ? ORDER BY request_id",
            [kind, status],
        )

    def judge_requests_for(self, parent_id: uuid.UUID, *, kind: str) -> dict[int, dict[str, Any]]:
        """Judge requests with a response for a generate request, by sample index."""
        judged: dict[int, dict[str, Any]] = {}
        for req in self._request_rows(
            "SELECT * FROM requests WHERE kind = ? AND parent_request_id = ? "
            "AND status IN ('received', 'processed') ORDER BY request_id",
            [kind, parent_id],
        ):
            index = req["response"].get("item_index")
            if isinstance(index, int):
                judged[index] = req
        return judged

    def requested_counts(
        self, kind: str, *, prompt_template: str, model_id: str
    ) -> dict[uuid.UUID, int]:
        """Samples requested per criterion with this template and model, API
        errors excluded (the attempt budget)."""
        return dict(
            self.con.execute(
                "SELECT combination_id, sum(requested_count) FROM requests "
                "WHERE kind = ? AND status != 'api_error' AND combination_id IS NOT NULL "
                "AND prompt_template = ? AND model_id = ? GROUP BY 1",
                [kind, prompt_template, model_id],
            ).fetchall()
        )

    def total_cost(self) -> float:
        return self.con.execute("SELECT coalesce(sum(cost_usd), 0) FROM requests").fetchone()[0]

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

    # ---------------------------------------------------------------- export / import

    def export(self, out_dir: Path) -> list[Path]:
        """Write utterances.csv (every sample, ordered by id); return it if it changed."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / EXPORT_FILE
        select = ", ".join(
            f"to_json({c}) AS {c}" if kind.endswith("[]") else c for c, kind in UTTERANCES.columns
        )
        with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
            staged = Path(tmp) / EXPORT_FILE
            self.con.execute(
                f"COPY (SELECT {select} FROM utterances ORDER BY utterance_id) "
                f"TO {_literal(staged)} (FORMAT csv, HEADER)"
            )
            if target.is_file() and target.read_bytes() == staged.read_bytes():
                return []
            shutil.copyfile(staged, target)
        return [target]

    def import_exports(self, export_dir: Path) -> None:
        """Rebuild an empty database's utterances from utterances.csv."""
        source = Path(export_dir) / EXPORT_FILE
        if self.count("utterances"):
            raise RefusedUpdate("import needs an empty database")
        if not source.is_file():
            return
        types = ", ".join(f"'{c}': 'VARCHAR'" for c in UTTERANCES.column_names)
        select = ", ".join(
            f"CAST(CAST({c} AS JSON) AS {kind}) AS {c}"
            if kind.endswith("[]")
            else f"CAST({c} AS {kind}) AS {c}"
            for c, kind in UTTERANCES.columns
        )
        self.con.execute(
            f"INSERT INTO utterances SELECT {select} FROM read_csv({_literal(source)}, "
            f"header = true, columns = {{{types}}}, allow_quoted_nulls = false)"
        )
