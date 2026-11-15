"""Human annotations kept separately from the immutable challenge dataset."""

import csv
import fcntl
import hashlib
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class ReviewStore:
    def __init__(self, source: Path, destination: Path):
        self.source, self.destination = Path(source), Path(destination)
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with self.source.open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.ids = {r["case_id"] for r in self.rows}
        if not self.rows or len(self.ids) != len(self.rows):
            raise ValueError("Dataset must contain unique case IDs and at least one case")
        self._load()

    def _load(self):
        if not self.destination.exists():
            return {"schema_version": 1, "source_sha256": self.digest, "reviews": {}}
        state = json.loads(self.destination.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("source_sha256") != self.digest:
            raise ValueError("Review file belongs to a different dataset or schema")
        return state

    def snapshot(self):
        reviews = self._load()["reviews"]
        return {
            "dataset": self.source.name,
            "cases": [
                {
                    "case_id": r["case_id"],
                    "text": r["text"],
                    "language": r["language"],
                    "review": reviews.get(r["case_id"]),
                }
                for r in self.rows
            ],
        }

    def save(self, case_id, decision, notes, revision):
        if case_id not in self.ids or decision not in {"backchannel", "barge_in"}:
            raise ValueError("Choose a valid case and decision")
        if not isinstance(notes, str) or len(notes) > 4000 or type(revision) is not int:
            raise ValueError("Invalid notes or revision")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load()
            previous = state["reviews"].get(case_id, {})
            if previous.get("revision", 0) != revision:
                raise ValueError("This review changed in another tab. Reload before saving.")
            record = {
                "decision": decision,
                "notes": notes,
                "revision": revision + 1,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            state["reviews"][case_id] = record
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.destination.parent,
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    temp_path = Path(f.name)
                    json.dump(state, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.destination)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            return record

    def export_csv(self):
        reviews = self._load()["reviews"]
        out = io.StringIO(newline="")
        writer = csv.writer(out)
        writer.writerow(["case_id", "text", "language", "decision", "notes", "updated_at"])
        for r in self.rows:
            review = reviews.get(r["case_id"], {})
            writer.writerow(
                [
                    r["case_id"],
                    r["text"],
                    r["language"],
                    review.get("decision", ""),
                    review.get("notes", ""),
                    review.get("updated_at", ""),
                ]
            )
        return out.getvalue()
