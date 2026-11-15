"""Fill-to-count run for the utterance track (docs/utterance-first-decision.md).

For every criterion of the plan file, in plan order:
need = required_count - accepted - in flight. Stored responses of an earlier,
interrupted run are processed first; a request sent but never answered is
marked api_error. Each response is validated; an utterance goes to the judge
only when the generator's label equals the intended label and its cleaned
text is new. The judge (Gemini Decisions API) asks one yes/no question per
criterion (utterance_criteria.judge_questions); the utterance is accepted only
when every answer is yes. Every sample becomes one row of `utterances`,
accepted or rejected with its reason.

Gemini calls (generation and judge) run on up to `concurrency` worker threads;
every store write happens on the calling thread, one response at a time, which
keeps deduplication exact. A generate request stays `received` until each of
its utterances has a verdict; a judge failure leaves it for the next run,
which reuses the verdicts already stored.
"""

import hashlib
import json
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
from barge_in_detector.generation.responses import UtteranceSample, parse_utterances
from barge_in_detector.generation.utterance_criteria import (
    JUDGE_KIND,
    JUDGE_TEMPLATE,
    JudgementError,
    judge_questions,
    parse_verdict,
)
from barge_in_detector.generation.utterances import UtteranceTrack, spec_of
from barge_in_detector.storage.store import Store

log = logging.getLogger(__name__)

KIND = prompts.UTTERANCE_REQUEST_KIND
_CHARS_PER_TOKEN = 4
_OUTPUT_TOKENS_PER_SAMPLE = 80  # estimate for --dry-run only; replace with measured values
_TEXT_FOR_ESTIMATE = "word " * 8


def text_hash_of(text: str) -> str:
    return hashlib.sha256(clean_text(text).encode()).hexdigest()


@dataclass(frozen=True)
class Job:
    target: dict[str, Any]
    count: int


@dataclass
class UtteranceRunReport:
    requests: int = 0
    judge_requests: int = 0
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


@dataclass
class _Screened:
    """A generate response whose surviving utterances wait for the judge."""

    req: dict[str, Any]
    target: dict[str, Any]
    rows: list[dict[str, Any]]  # sample rows already rejected
    candidates: list[tuple[UtteranceSample, str]]  # (sample, text hash)
    judged: dict[int, dict[str, Any]] = field(default_factory=dict)  # sample index -> request
    waiting: int = 0
    failed: bool = False


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
        targets: Sequence[Mapping[str, Any]],
        *,
        progress: Callable[[str], None] | None = None,
    ):
        """`targets`: the plan's criteria (utterances.plan_targets)."""
        self.store = store
        self.targets = list(targets)
        self.cat = cat
        self.track = track
        self.client = client
        self.config = config
        self.progress = progress or (lambda message: log.info("%s", message))
        self.system = prompts.render_utterance_system(cat, track)
        self._seen: set[str] = set()  # hashes of accepted utterances
        self._judging: set[str] = set()  # hashes of utterances waiting for the judge

    def _spec(self, target: Mapping[str, Any]) -> dict[str, str]:
        return spec_of(target, self.track.factors)

    # ---------------------------------------------------------------- jobs

    def _plan_jobs(
        self, targets: Sequence[Mapping[str, Any]], limit: int | None
    ) -> tuple[list[Job], list[dict[str, Any]]]:
        """Jobs for every short criterion. The attempt budget counts requests
        made with the current prompt template and generator model only: a new
        prompt version or model starts a fresh budget."""
        accepted = self.store.accepted_counts()
        requested = self.store.requested_counts(
            KIND,
            prompt_template=prompts.UTTERANCE_TEMPLATE,
            model_id=self.config.generator_model,
        )
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
        export_dir: Path | None = None,
    ) -> UtteranceRunReport:
        prices = self.config._prices(self.config.generator_model)
        priced = all(prices.get(k) is not None for k in ("input", "cached_input", "output"))
        if budget_usd is not None and not priced:
            raise ValueError("--budget-usd needs generator prices in generation/config.yaml")
        report = UtteranceRunReport(cost_usd=0.0 if priced else None)
        targets = [t for t in self.targets if not languages or t["language"] in languages]
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

        self._seen = self.store.accepted_text_hashes()
        self._judging = set()
        try:
            resumed = self._resume()
            self._execute_all(targets, resumed, limit, budget_usd, report)
            accepted = self.store.accepted_counts()
            _, exhausted = self._plan_jobs(targets, None)
            report.exhausted = [_describe(t) for t in exhausted]
            report.short = [
                _describe(t)
                for t in targets
                if t["required_count"] > accepted.get(t["combination_id"], 0)
            ]
        finally:
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
        judge_input = 0
        for job in jobs:
            text, questions = judge_questions(
                self.cat, self._spec(job.target), _TEXT_FOR_ESTIMATE, None
            )
            size = len(text) + len(json.dumps(questions, ensure_ascii=False))
            judge_input += job.count * size // _CHARS_PER_TOKEN
        usage = {
            "input_tokens": fresh + system * len(jobs),
            "cached_input_tokens": 0,
            "output_tokens": report.planned_samples * _OUTPUT_TOKENS_PER_SAMPLE,
        }
        judge = {"input_tokens": judge_input, "cached_input_tokens": 0, "output_tokens": 0}
        report.estimated_tokens = {
            "generator_input": usage["input_tokens"],
            "generator_output": usage["output_tokens"],
            "judge_input": judge_input,
        }
        costs = [
            self.config.cost(usage, self.config.generator_model),
            self.config.cost(judge, self.config.judge_model),
        ]
        report.estimated_cost_usd = None if None in costs else sum(costs)
        report.exhausted = [_describe(t) for t in exhausted]

    # ---------------------------------------------------------------- requests

    def _resume(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Fail requests sent but never answered; return stored responses to finish."""
        for kind in (KIND, JUDGE_KIND):
            for req in self.store.requests(kind=kind, status="submitted"):
                self.store.fail_request(
                    req["request_id"], status="api_error", error="interrupted before a response"
                )
        targets = {t["combination_id"]: t for t in self.targets}
        resumed = []
        for req in self.store.requests(kind=KIND, status="received"):
            target = targets.get(req["combination_id"])
            if target is not None:
                self.progress(f"resuming stored response for {_describe(target)}")
                resumed.append((req, target))
        return resumed

    def _execute_all(
        self,
        targets: Sequence[Mapping[str, Any]],
        resumed: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        limit: int | None,
        budget_usd: float | None,
        report: UtteranceRunReport,
    ) -> None:
        """Up to `concurrency` Gemini calls in flight; every store write stays here.

        Requests are inserted before their call starts, so they count as in
        flight when jobs are planned again (after rejections).
        """
        workers = max(1, self.config.concurrency)
        queue: list[Job] = []
        running: dict[Future, tuple[Any, ...]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for req, target in resumed:
                self._screen(req, target, report, pool, running)
            while True:
                while len(running) < workers:
                    if limit is not None and report.requests >= limit:
                        break
                    if budget_usd is not None and (report.cost_usd or 0) >= budget_usd:
                        break
                    if not queue:
                        if running:
                            break  # plan again only once the work in flight is done
                        remaining = None if limit is None else limit - report.requests
                        queue, _ = self._plan_jobs(targets, remaining)
                        if not queue:
                            break
                    job = queue.pop(0)
                    report.requests += 1
                    request_id, user = self._insert(job)
                    running[pool.submit(self._generate, user)] = ("generate", job, request_id)
                if not running:
                    return
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    kind, *info = running.pop(future)
                    if kind == "generate":
                        job, request_id = info
                        req = self._record_generation(request_id, future, report)
                        if req is not None:
                            self._screen(req, job.target, report, pool, running)
                        self.progress(
                            f"[{report.requests}] {job.target['language']} "
                            f"{job.target['caller_intent']}: {report.accepted} accepted so far"
                        )
                    else:
                        screened, sample, judge_id = info
                        self._record_verdict(screened, sample, judge_id, future, report)

    def _insert(self, job: Job) -> tuple[Any, str]:
        target = job.target
        user = prompts.render_utterance_request(self.cat, self._spec(target), count=job.count)
        request_id = self.store.insert_request(
            kind=KIND,
            combination_id=target["combination_id"],
            language=target["language"],
            requested_count=job.count,
            model_id=self.config.generator_model,
            prompt_template=prompts.UTTERANCE_TEMPLATE,
            prompt_hash=prompts.prompt_hash(self.system, user),
        )
        return request_id, user

    def _generate(self, user: str) -> Any:
        """Runs on a worker thread: the Gemini call only, no store access."""
        return self.client.generate(
            model=self.config.generator_model,
            system_prompt=self.system,
            user_prompt=user,
            schema=prompts.UTTERANCE_SCHEMA,
            thinking_level=self.config.thinking_level,
        )

    def _judge(self, judge_input: str, questions: Mapping[str, Any]) -> Any:
        """Runs on a worker thread: the Decisions API call only."""
        return self.client.decide(
            model=self.config.judge_model, input=judge_input, questions=questions
        )

    def _add_cost(self, report: UtteranceRunReport, cost: float | None) -> None:
        if report.cost_usd is not None:
            report.cost_usd += cost or 0.0

    def _record_generation(
        self, request_id: Any, future: Future, report: UtteranceRunReport
    ) -> dict[str, Any] | None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            self.store.fail_request(request_id, status="api_error", error=redact(str(exc)))
            report.api_errors += 1
            return None
        cost = self.config.cost(result.usage, self.config.generator_model)
        self.store.record_response(
            request_id, response=result.response, usage=result.usage, cost_usd=cost
        )
        self._add_cost(report, cost)
        return self.store.request(request_id)

    # ---------------------------------------------------------------- validation

    def _row(
        self,
        req: Mapping[str, Any],
        index: int,
        raw: Any,
        *,
        status: str,
        reason: str | None = None,
        text_hash: str | None = None,
        judge_req: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One sample row: the raw sample as Gemini wrote it, and the verdict."""
        sample = raw if isinstance(raw, dict) else {}
        text = sample.get("text") if isinstance(sample.get("text"), str) else None
        return {
            "status": status,
            "reason": reason,
            "text": text,
            "label": sample.get("label") if isinstance(sample.get("label"), str) else None,
            "raw_sample": json.dumps(raw, ensure_ascii=False),
            "judge_answers": (
                judge_req["response"].get("answers") if judge_req is not None else None
            ),
            "text_hash": text_hash or (text_hash_of(text) if text else None),
            "request_id": req["request_id"],
            "judge_request_id": judge_req["request_id"] if judge_req is not None else None,
            "sample_index": index,
            "model_id": req["model_id"],
            "prompt_template": req["prompt_template"],
            "judge_model_id": judge_req["model_id"] if judge_req is not None else None,
            "judge_template": judge_req["prompt_template"] if judge_req is not None else None,
        }

    def _reject(
        self,
        report: UtteranceRunReport,
        stage: str,
        reason: str,
        keys: Sequence[str] | None = None,
    ) -> str:
        report.rejections[stage] += 1
        for key in keys or [reason.split(" ")[0].split("=")[0]]:
            report.reasons[f"{stage}:{key}"] += 1
        return f"{stage}: {reason}"

    def _screen(
        self,
        req: dict[str, Any],
        target: dict[str, Any],
        report: UtteranceRunReport,
        pool: ThreadPoolExecutor,
        running: dict[Future, tuple[Any, ...]],
    ) -> None:
        """Validate a stored response, then send each surviving utterance to the judge."""
        text = req["response"]["text"]
        parsed = parse_utterances(
            self.cat,
            text,
            spec=self._spec(target),
            count=req["requested_count"],
            labels=self.track.labels,
        )
        if parsed.fatal:
            self._reject(report, "schema", parsed.fatal)
            self.store.fail_request(req["request_id"], status="schema_invalid", error=parsed.fatal)
            return
        raw_samples = json.loads(text)["samples"]
        rows = [
            self._row(
                req,
                p.index,
                raw_samples[p.index],
                status="rejected",
                reason=self._reject(report, "schema", p.reason),
            )
            for p in parsed.problems
        ]
        candidates = []
        for sample in parsed.samples:
            if sample.label != target["label"]:
                reason = f"intended={target['label']} gemini={sample.label}"
                rows.append(
                    self._row(
                        req,
                        sample.index,
                        sample.data,
                        status="rejected",
                        reason=self._reject(report, "label_mismatch", reason),
                    )
                )
                continue
            h = text_hash_of(sample.data["text"])
            if h in self._seen or h in self._judging:
                rows.append(
                    self._row(
                        req,
                        sample.index,
                        sample.data,
                        status="rejected",
                        reason=self._reject(report, "dedupe", "duplicate of an accepted utterance"),
                        text_hash=h,
                    )
                )
                continue
            self._judging.add(h)
            candidates.append((sample, h))
        screened = _Screened(req, target, rows, candidates)
        stored = self.store.judge_requests_for(req["request_id"], kind=JUDGE_KIND)
        for sample, _ in candidates:
            if sample.index in stored:
                screened.judged[sample.index] = stored[sample.index]
                continue
            judge_input, questions = judge_questions(
                self.cat, self._spec(target), sample.data["text"]
            )
            judge_id = self.store.insert_request(
                kind=JUDGE_KIND,
                parent_request_id=req["request_id"],
                combination_id=target["combination_id"],
                language=target["language"],
                requested_count=1,
                model_id=self.config.judge_model,
                prompt_template=JUDGE_TEMPLATE,
                prompt_hash=prompts.prompt_hash(
                    judge_input, json.dumps(questions, sort_keys=True, ensure_ascii=False)
                ),
            )
            report.judge_requests += 1
            screened.waiting += 1
            future = pool.submit(self._judge, judge_input, questions)
            running[future] = ("judge", screened, sample, judge_id)
        if screened.waiting == 0:
            self._finish(screened, report)

    def _record_verdict(
        self,
        screened: _Screened,
        sample: UtteranceSample,
        judge_id: Any,
        future: Future,
        report: UtteranceRunReport,
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            self.store.fail_request(judge_id, status="api_error", error=redact(str(exc)))
            report.api_errors += 1
            screened.failed = True
        else:
            cost = self.config.cost(result.usage, self.config.judge_model)
            self.store.record_response(
                judge_id,
                response={"item_index": sample.index, **result.response},
                usage=result.usage,
                cost_usd=cost,
            )
            self._add_cost(report, cost)
            screened.judged[sample.index] = self.store.request(judge_id)
        screened.waiting -= 1
        if screened.waiting == 0:
            self._finish(screened, report)

    def _finish(self, screened: _Screened, report: UtteranceRunReport) -> None:
        """Write every sample row of one response, in one transaction."""
        req, target = screened.req, screened.target
        verdicts = {}
        if not screened.failed:
            for sample, _ in screened.candidates:
                judge_req = screened.judged[sample.index]
                _, questions = judge_questions(self.cat, self._spec(target), sample.data["text"])
                try:
                    verdicts[sample.index] = parse_verdict(judge_req["response"], list(questions))
                except JudgementError as exc:
                    self.store.fail_request(
                        judge_req["request_id"], status="schema_invalid", error=str(exc)
                    )
                    screened.failed = True
        for _, h in screened.candidates:
            self._judging.discard(h)
        if screened.failed:
            return  # the response stays received; the next run judges what is missing
        rows = list(screened.rows)
        accepted = 0
        for sample, h in screened.candidates:
            verdict = verdicts[sample.index]
            judge_req = screened.judged[sample.index]
            if verdict.failed:
                reason = self._reject(report, "judge", verdict.reason(), verdict.failed)
                rows.append(
                    self._row(
                        req,
                        sample.index,
                        sample.data,
                        status="rejected",
                        reason=reason,
                        text_hash=h,
                        judge_req=judge_req,
                    )
                )
                continue
            rows.append(
                self._row(
                    req,
                    sample.index,
                    sample.data,
                    status="accepted",
                    text_hash=h,
                    judge_req=judge_req,
                )
            )
            self._seen.add(h)
            accepted += 1
        rows.sort(key=lambda r: r["sample_index"])
        with self.store.transaction():
            for row in rows:
                self.store.insert_utterance(target, row)
            self.store.mark_processed(req["request_id"])
            for judge_req in screened.judged.values():
                if judge_req["status"] == "received":
                    self.store.mark_processed(judge_req["request_id"])
        report.accepted += accepted
