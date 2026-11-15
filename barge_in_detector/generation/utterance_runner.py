"""Fill-to-count run for the utterance track (docs/utterance-first-decision.md).

For every active utterance criterion, in plan order:
need = required_count - accepted - in flight. Stored responses of an earlier,
interrupted run are processed first; a request sent but never answered is
marked api_error. Each response is validated and every utterance is accepted
only when the generator's label equals the intended label and its cleaned text
is new. There is no judge pass in this version (user decision, 2026-11-15).
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from barge_in_detector.catalogue import Catalogue
from barge_in_detector.cleaner import clean_text
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import LlmClient, redact
from barge_in_detector.generation.responses import parse_utterances
from barge_in_detector.generation.utterances import UtteranceTrack, spec_of
from barge_in_detector.storage.store import Store

log = logging.getLogger(__name__)

KIND = prompts.UTTERANCE_REQUEST_KIND
_CHARS_PER_TOKEN = 4
_OUTPUT_TOKENS_PER_SAMPLE = 80  # estimate for --dry-run only; replace with measured values


def text_hash(text: str) -> str:
    return hashlib.sha256(clean_text(text).encode()).hexdigest()


@dataclass(frozen=True)
class Job:
    target: dict[str, Any]
    count: int


@dataclass
class UtteranceRunReport:
    run_id: Any = None
    requests: int = 0
    accepted: int = 0
    rejections: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    api_errors: int = 0
    cost_usd: float | None = None
    exhausted: list[str] = field(default_factory=list)
    short: list[str] = field(default_factory=list)
    planned_requests: int = 0
    planned_samples: int = 0
    planned_by_label: Counter = field(default_factory=Counter)
    estimated_tokens: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float | None = None


def _describe(target: Mapping[str, Any]) -> str:
    return f"{target['language']} {target['caller_intent']} {target['combination_id']}"


class UtteranceRunner:
    def __init__(
        self,
        store: Store,
        cat: Catalogue,
        track: UtteranceTrack,
        client: LlmClient,
        config: GenerationConfig,
        *,
        progress: Callable[[str], None] | None = None,
    ):
        self.store = store
        self.cat = cat
        self.track = track
        self.client = client
        self.config = config
        self.progress = progress or (lambda message: log.info("%s", message))
        self.system = prompts.render_utterance_system(cat, track)

    def _spec(self, target: Mapping[str, Any]) -> dict[str, str]:
        return spec_of(target, self.track.factors)

    # ---------------------------------------------------------------- jobs

    def _plan_jobs(
        self, targets: Sequence[Mapping[str, Any]], limit: int | None
    ) -> tuple[list[Job], list[dict[str, Any]]]:
        accepted = self.store.utterance_accepted_counts()
        requested = self.store.requested_counts(KIND)
        in_flight = {
            r["combination_id"]: r["requested_count"]
            for status in ("submitted", "received")
            for r in self.store.requests(kind=KIND, status=status)
        }
        jobs: list[Job] = []
        exhausted = []
        for t in targets:
            cid = t["combination_id"]
            need = t["required_count"] - accepted.get(cid, 0) - in_flight.get(cid, 0)
            if need <= 0:
                continue
            budget = self.config.max_attempts_per_sample * t["required_count"]
            remaining = budget - requested.get(cid, 0)
            if remaining <= 0:
                exhausted.append(t)
                continue
            while need > 0 and remaining > 0:
                n = min(need, self.config.batch_size, remaining)
                jobs.append(Job(t, n))
                need -= n
                remaining -= n
        if limit is not None:
            jobs = jobs[: max(limit, 0)]
        return jobs, exhausted

    # ---------------------------------------------------------------- run

    def run(
        self,
        *,
        languages: Sequence[str] | None = None,
        limit: int | None = None,
        budget_usd: float | None = None,
        dry_run: bool = False,
        run_name: str = "generate-utterances",
        export_dir: Path | None = None,
    ) -> UtteranceRunReport:
        prices = self.config._prices(self.config.generator_model)
        priced = all(prices.get(k) is not None for k in ("input", "cached_input", "output"))
        if budget_usd is not None and not priced:
            raise ValueError("--budget-usd needs generator prices in generation/config.yaml")
        report = UtteranceRunReport(cost_usd=0.0 if priced else None)
        targets = self.store.utterance_targets(languages=languages)
        pending = self.store.requests(kind=KIND, status="submitted") + self.store.requests(
            kind=KIND, status="received"
        )
        jobs, exhausted = self._plan_jobs(targets, limit)
        if dry_run:
            self._estimate(report, jobs, exhausted)
            return report
        if not jobs and not pending:
            report.exhausted = [_describe(t) for t in exhausted]
            return report

        run_id = self.store.start_run(
            name=run_name,
            revision_id=None,
            settings={
                "track": "utterance",
                "languages": list(languages or []),
                "limit": limit,
                "budget_usd": budget_usd,
                "batch_size": self.config.batch_size,
                "concurrency": self.config.concurrency,
                "max_attempts_per_sample": self.config.max_attempts_per_sample,
                "generator_template": prompts.UTTERANCE_TEMPLATE,
                "judge": None,
            },
            generator_model_id=self.config.generator_model,
            judge_model_id=None,
        )
        report.run_id = run_id
        try:
            self._resume(run_id, report)
            self._execute_all(run_id, targets, limit, budget_usd, report)
            accepted = self.store.utterance_accepted_counts()
            _, exhausted = self._plan_jobs(targets, None)
            report.exhausted = [_describe(t) for t in exhausted]
            report.short = [
                _describe(t)
                for t in targets
                if t["required_count"] > accepted.get(t["combination_id"], 0)
            ]
        finally:
            self.store.finish_run(run_id)
            self.client.close()
        if export_dir is not None:
            self.store.export(export_dir)
        return report

    def _estimate(
        self, report: UtteranceRunReport, jobs: Sequence[Job], exhausted: Sequence[Mapping]
    ) -> None:
        report.planned_requests = len(jobs)
        report.planned_samples = sum(j.count for j in jobs)
        for job in jobs:
            report.planned_by_label[f"{job.target['language']}:{job.target['label']}"] += job.count
        system = len(self.system) // _CHARS_PER_TOKEN
        fresh = sum(
            len(prompts.render_utterance_request(self.cat, self._spec(j.target), count=j.count))
            // _CHARS_PER_TOKEN
            for j in jobs
        )
        usage = {
            "input_tokens": fresh + system * len(jobs),
            "cached_input_tokens": 0,
            "output_tokens": report.planned_samples * _OUTPUT_TOKENS_PER_SAMPLE,
        }
        report.estimated_tokens = {
            "generator_input": usage["input_tokens"],
            "generator_output": usage["output_tokens"],
        }
        report.estimated_cost_usd = self.config.cost(usage, self.config.generator_model)
        report.exhausted = [_describe(t) for t in exhausted]

    # ---------------------------------------------------------------- requests

    def _execute_all(
        self,
        run_id: Any,
        targets: Sequence[Mapping[str, Any]],
        limit: int | None,
        budget_usd: float | None,
        report: UtteranceRunReport,
    ) -> None:
        """Up to `concurrency` Gemini calls in flight; every DB write stays here.

        Requests are inserted before their call starts, so they count as in
        flight when jobs are planned again (after rejections). Responses are
        validated one at a time on this thread, which keeps deduplication exact.
        """
        workers = max(1, self.config.concurrency)
        queue: list[Job] = []
        running: dict[Future, tuple[Job, Any, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while len(running) < workers:
                    if limit is not None and report.requests >= limit:
                        break
                    if budget_usd is not None and (report.cost_usd or 0) >= budget_usd:
                        break
                    if not queue:
                        if running:
                            break  # re-plan only once the requests in flight are processed
                        remaining = None if limit is None else limit - report.requests
                        queue, _ = self._plan_jobs(targets, remaining)
                        if not queue:
                            break
                    job = queue.pop(0)
                    report.requests += 1
                    request_id, user = self._insert(run_id, job)
                    running[pool.submit(self._call, user)] = (job, request_id, user)
                if not running:
                    return
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    job, request_id, _user = running.pop(future)
                    self._finish(run_id, job, request_id, future, report)
                    self.progress(
                        f"[{report.requests}] {job.target['language']} "
                        f"{job.target['caller_intent']}: {report.accepted} accepted so far"
                    )

    def _insert(self, run_id: Any, job: Job) -> tuple[Any, str]:
        target = job.target
        user = prompts.render_utterance_request(self.cat, self._spec(target), count=job.count)
        request_id = self.store.insert_request(
            run_id=run_id,
            kind=KIND,
            combination_id=target["combination_id"],
            contrastive_group_id=None,
            language=target["language"],
            requested_count=job.count,
            model_id=self.config.generator_model,
            prompt_template=prompts.UTTERANCE_TEMPLATE,
            prompt_hash=prompts.prompt_hash(self.system, user),
        )
        return request_id, user

    def _call(self, user: str) -> Any:
        """Runs on a worker thread: the Gemini call only, no store access."""
        return self.client.generate(
            model=self.config.generator_model,
            system_prompt=self.system,
            user_prompt=user,
            schema=prompts.UTTERANCE_SCHEMA,
            thinking_level=self.config.thinking_level,
        )

    def _finish(
        self, run_id: Any, job: Job, request_id: Any, future: Future, report: UtteranceRunReport
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            self.store.fail_request(request_id, status="api_error", error=redact(str(exc)))
            report.api_errors += 1
            return
        cost = self.config.cost(result.usage, self.config.generator_model)
        self.store.record_response(
            request_id, response=result.response, usage=result.usage, cost_usd=cost
        )
        if report.cost_usd is not None:
            report.cost_usd += cost or 0.0
        self._process(run_id, self.store.request(request_id), job.target, report)

    def _resume(self, run_id: Any, report: UtteranceRunReport) -> None:
        """Finish earlier work before any new request."""
        for req in self.store.requests(kind=KIND, status="submitted"):
            self.store.fail_request(
                req["request_id"], status="api_error", error="interrupted before a response"
            )
        targets = {
            t["combination_id"]: t
            for status in ("active", "retired")
            for t in self.store.utterance_targets(status=status)
        }
        for req in self.store.requests(kind=KIND, status="received"):
            target = targets.get(req["combination_id"])
            if target is None:
                continue
            self.progress(f"resuming stored response for {_describe(target)}")
            self._process(run_id, req, target, report)

    def _process(
        self,
        run_id: Any,
        req: Mapping[str, Any],
        target: Mapping[str, Any],
        report: UtteranceRunReport,
    ) -> None:
        request_id = req["request_id"]
        parsed = parse_utterances(
            self.cat,
            req["response"]["text"],
            spec=self._spec(target),
            count=req["requested_count"],
            labels=self.track.labels,
        )

        def rejection(index, stage, reason, label):
            report.rejections[stage] += 1
            report.reasons[f"{stage}:{reason.split(' ')[0].split('=')[0]}"] += 1
            return {
                "request_id": request_id,
                "combination_id": target["combination_id"],
                "language": target["language"],
                "run_id": run_id,
                "sample_index": index,
                "stage": stage,
                "reason": reason,
                "gemini_label": label,
                "judge_label": None,
            }

        if parsed.fatal:
            with self.store.transaction():
                self.store.insert_rejection(rejection(None, "schema", parsed.fatal, None))
                self.store.fail_request(request_id, status="schema_invalid", error=parsed.fatal)
            return

        rejections = [rejection(p.index, "schema", p.reason, p.label) for p in parsed.problems]
        seen = self.store.utterance_text_hashes()
        accepted = []
        for sample in parsed.samples:
            if sample.label != target["label"]:
                rejections.append(
                    rejection(
                        sample.index,
                        "label_mismatch",
                        f"intended={target['label']} gemini={sample.label}",
                        sample.label,
                    )
                )
                continue
            h = text_hash(sample.data["text"])
            if h in seen:
                rejections.append(
                    rejection(
                        sample.index, "dedupe", "duplicate of an accepted utterance", sample.label
                    )
                )
                continue
            seen.add(h)
            accepted.append(
                {
                    "request_id": request_id,
                    "run_id": run_id,
                    "sample_index": sample.index,
                    "text": sample.data["text"],
                    "gemini_label": sample.label,
                    "gemini_rationale": sample.data["rationale"],
                    "barge_in_from_word": sample.barge_in_from_word,
                    "judge_label": None,
                    "judge_model_id": None,
                    "model_id": req["model_id"],
                    "prompt_template": req["prompt_template"],
                    "text_hash": h,
                }
            )
        with self.store.transaction():
            for row in accepted:
                self.store.insert_utterance(target, row)
            for row in rejections:
                self.store.insert_rejection(row)
            self.store.mark_processed(request_id)
        report.accepted += len(accepted)
