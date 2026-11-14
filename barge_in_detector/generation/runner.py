"""Fill-to-count generation run (docs/data-format.md#generation-fill-each-criterion-to-its-required-count).

For every active criterion, in plan order:
need = required_count - accepted (in-flight requests are resolved first).
A criterion (or contrastive group) is skipped when need <= 0 or its attempt
budget is used up. Requests carry at most `batch_size` samples. Each response
is validated, judged blind in one judge call, deduplicated and accepted only
when the generator's label, the judge's label and the intended label agree.
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
from barge_in_detector.cleaner import clean_window
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import LlmClient, redact
from barge_in_detector.generation.responses import (
    JudgementError,
    Sample,
    parse_generation,
    parse_judgement,
)
from barge_in_detector.storage.schema import SPEC_FACTORS
from barge_in_detector.storage.store import Store

log = logging.getLogger(__name__)

# Rough sizes for --dry-run estimates only.
_CHARS_PER_TOKEN = 4
_OUTPUT_TOKENS_PER_SAMPLE = 160
_JUDGE_OUTPUT_TOKENS_PER_ITEM = 60


@dataclass
class Unit:
    """One criterion, or one contrastive group instance (members in role order)."""

    key: str
    targets: list[dict[str, Any]]
    group: dict[str, Any] | None

    @property
    def members(self) -> int:
        return len(self.targets)

    @property
    def language(self) -> str:
        return self.targets[0]["language"]

    @property
    def specs(self) -> list[dict[str, str]]:
        return [{f: t[f] for f in SPEC_FACTORS} for t in self.targets]

    def describe(self) -> str:
        intents = "/".join(t["caller_intent"] for t in self.targets)
        return f"{self.language} {intents}"


@dataclass(frozen=True)
class Job:
    unit: Unit
    count: int  # samples, or sets for a contrastive group

    @property
    def samples(self) -> int:
        return self.count * self.unit.members


@dataclass
class Outcome:
    accepted: int = 0
    rejections: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0
    api_errors: int = 0
    responses: int = 0


@dataclass
class RunReport:
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

    def add(self, outcome: Outcome) -> None:
        self.accepted += outcome.accepted
        self.rejections.update(outcome.rejections)
        self.reasons.update(outcome.reasons)
        self.api_errors += outcome.api_errors
        if self.cost_usd is not None:
            self.cost_usd += outcome.cost_usd


def text_hash(sample: Sample) -> str:
    d = sample.data
    window = clean_window(
        [
            (d["context_turn"]["speaker"], d["context_turn"]["text"]),
            (d["agent_speaker"], d["agent_spoken_text"]),
            (d["candidate_speaker"], d["candidate_text"]),
        ]
    )
    return hashlib.sha256(window.transcript.encode()).hexdigest()


class Runner:
    def __init__(
        self,
        store: Store,
        cat: Catalogue,
        client: LlmClient,
        config: GenerationConfig,
        *,
        progress: Callable[[str], None] | None = None,
    ):
        self.store = store
        self.cat = cat
        self.client = client
        self.config = config
        self.progress = progress or (lambda message: log.info("%s", message))
        self.generator_system = prompts.render_generator_system(cat)
        self.judge_system = prompts.render_judge_system(cat)
        self._groups = {g["id"]: g for g in cat.contrastive}

    # ---------------------------------------------------------------- units and jobs

    def _units(self, targets: Sequence[Mapping[str, Any]]) -> list[Unit]:
        units: dict[str, Unit] = {}
        for t in targets:
            group_id = t["contrastive_group_id"]
            if group_id:
                group = self._groups[group_id.split(":", 1)[0]]
                units.setdefault(group_id, Unit(group_id, [], group)).targets.append(dict(t))
            else:
                units[str(t["combination_id"])] = Unit(str(t["combination_id"]), [dict(t)], None)
        for unit in units.values():
            unit.targets.sort(key=lambda t: t["contrastive_role"] or 0)
        return list(units.values())

    def _plan_jobs(self, units: Sequence[Unit], limit: int | None) -> tuple[list[Job], list[Unit]]:
        accepted = self.store.accepted_counts()
        by_criterion, by_group = self.store.attempts()
        flight_criterion, flight_group = self.store.in_flight()
        jobs: list[Job] = []
        exhausted: list[Unit] = []
        capacity_default = self.config.batch_size
        for unit in units:
            in_flight = (
                flight_group.get(unit.key, 0) // unit.members
                if unit.group
                else flight_criterion.get(unit.targets[0]["combination_id"], 0)
            )
            need = (
                max(
                    t["required_count"] - accepted.get(t["combination_id"], 0) for t in unit.targets
                )
                - in_flight
            )
            if need <= 0:
                continue
            required = max(t["required_count"] for t in unit.targets)
            budget = self.config.max_attempts_per_sample * required * unit.members
            used = (
                by_group.get(unit.key, 0)
                if unit.group
                else by_criterion.get(unit.targets[0]["combination_id"], 0)
            )
            remaining = (budget - used) // unit.members
            if remaining <= 0:
                exhausted.append(unit)
                continue
            capacity = max(1, capacity_default // unit.members)
            while need > 0 and remaining > 0:
                n = min(need, capacity, remaining)
                jobs.append(Job(unit, n))
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
        run_name: str = "generate",
        export_dir: Path | None = None,
    ) -> RunReport:
        if budget_usd is not None and not self.config.has_prices:
            raise ValueError("--budget-usd needs prices in generation/config.yaml")
        report = RunReport(cost_usd=0.0 if self.config.has_prices else None)
        units = self._units(self.store.targets(languages=languages))
        all_units = self._units(
            self.store.targets(status="active") + self.store.targets(status="retired")
        )
        pending = (
            self.store.requests(kind="generate", status="submitted")
            + self.store.requests(kind="generate", status="received")
            + self.store.requests(kind="judge", status="submitted")
        )
        jobs, exhausted = self._plan_jobs(units, limit)
        if dry_run:
            self._estimate(report, jobs, exhausted)
            return report
        if not jobs and not pending:
            report.exhausted = [u.describe() for u in exhausted]
            return report

        run_id = self.store.start_run(
            name=run_name,
            revision_id=None,
            settings={
                "languages": list(languages or []),
                "limit": limit,
                "budget_usd": budget_usd,
                "batch_size": self.config.batch_size,
                "max_attempts_per_sample": self.config.max_attempts_per_sample,
                "concurrency": self.config.concurrency,
                "generator_template": prompts.GENERATOR_TEMPLATE,
                "judge_template": prompts.JUDGE_TEMPLATE,
            },
            generator_model_id=self.config.generator_model,
            judge_model_id=self.config.judge_model,
        )
        report.run_id = run_id
        try:
            self._resume(run_id, all_units, report)
            while True:
                remaining = None if limit is None else limit - report.requests
                if remaining is not None and remaining <= 0:
                    break
                jobs, exhausted = self._plan_jobs(units, remaining)
                if not jobs:
                    break
                responses = self._execute_all(run_id, jobs, report, budget_usd)
                if responses == 0 or (
                    budget_usd is not None and (report.cost_usd or 0) >= budget_usd
                ):
                    break
            accepted = self.store.accepted_counts()
            _, exhausted = self._plan_jobs(units, None)
            report.exhausted = [u.describe() for u in exhausted]
            report.short = [
                u.describe()
                for u in units
                if any(
                    t["required_count"] > accepted.get(t["combination_id"], 0) for t in u.targets
                )
            ]
        finally:
            self.store.finish_run(run_id)
            self.client.close()
        if export_dir is not None:
            self.store.export(export_dir)
        return report

    def _estimate(self, report: RunReport, jobs: Sequence[Job], exhausted: Sequence[Unit]) -> None:
        report.planned_requests = len(jobs)
        report.planned_samples = sum(j.samples for j in jobs)
        for job in jobs:
            for t in job.unit.targets:
                report.planned_by_label[f"{t['language']}:{t['label']}"] += job.count
        system = len(self.generator_system) // _CHARS_PER_TOKEN
        judge_system = len(self.judge_system) // _CHARS_PER_TOKEN
        fresh = sum(
            len(
                prompts.render_generation_request(
                    self.cat, j.unit.specs, count=j.count, group=j.unit.group
                )
            )
            // _CHARS_PER_TOKEN
            for j in jobs
        )
        samples = report.planned_samples
        report.estimated_tokens = {
            "input": fresh + 120 * samples,  # judge items are ~120 tokens each
            "cached_input": (system + judge_system) * len(jobs),
            "output": samples * (_OUTPUT_TOKENS_PER_SAMPLE + _JUDGE_OUTPUT_TOKENS_PER_ITEM),
        }
        report.estimated_cost_usd = self.config.cost(
            {
                "input_tokens": report.estimated_tokens["input"]
                + report.estimated_tokens["cached_input"],
                "cached_input_tokens": report.estimated_tokens["cached_input"],
                "output_tokens": report.estimated_tokens["output"],
            }
        )
        report.exhausted = [u.describe() for u in exhausted]

    def _execute_all(
        self, run_id: Any, jobs: Sequence[Job], report: RunReport, budget_usd: float | None
    ) -> int:
        responses = 0
        queue = list(jobs)
        running: set[Future] = set()
        done_count = 0
        with ThreadPoolExecutor(max_workers=max(1, self.config.concurrency)) as pool:
            while queue or running:
                over_budget = budget_usd is not None and (report.cost_usd or 0) >= budget_usd
                while queue and not over_budget and len(running) < max(1, self.config.concurrency):
                    job = queue.pop(0)
                    report.requests += 1
                    future = pool.submit(self._execute, run_id, job)
                    future.job = job  # type: ignore[attr-defined]
                    running.add(future)
                if over_budget:
                    queue.clear()
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    outcome = future.result()
                    report.add(outcome)
                    responses += outcome.responses
                    done_count += 1
                    job = future.job  # type: ignore[attr-defined]
                    rejected = sum(outcome.rejections.values())
                    self.progress(
                        f"[{done_count}/{len(jobs)}] {job.unit.describe()}: "
                        f"+{outcome.accepted} accepted, {rejected} rejected"
                        + (", API error" if outcome.api_errors else "")
                    )
        return responses

    # ---------------------------------------------------------------- one request

    def _execute(self, run_id: Any, job: Job) -> Outcome:
        st = self.store.cursor()
        try:
            unit = job.unit
            user = prompts.render_generation_request(
                self.cat, unit.specs, count=job.count, group=unit.group
            )
            request_id = st.insert_request(
                run_id=run_id,
                kind="generate",
                combination_id=None if unit.group else unit.targets[0]["combination_id"],
                contrastive_group_id=unit.key if unit.group else None,
                language=unit.language,
                requested_count=job.samples,
                model_id=self.config.generator_model,
                prompt_template=prompts.GENERATOR_TEMPLATE,
                prompt_hash=prompts.prompt_hash(self.generator_system, user),
            )
            try:
                result = self.client.generate(
                    model=self.config.generator_model,
                    system_prompt=self.generator_system,
                    user_prompt=user,
                    schema=prompts.GENERATOR_SCHEMA,
                    thinking_level=self.config.thinking_level,
                )
            except Exception as exc:  # noqa: BLE001 - every failure is recorded
                st.fail_request(request_id, status="api_error", error=redact(str(exc)))
                return Outcome(api_errors=1)
            cost = self.config.cost(result.usage)
            st.record_response(
                request_id, response=result.response, usage=result.usage, cost_usd=cost
            )
            outcome = self._process(st, run_id, st.request(request_id), unit)
            outcome.cost_usd += cost or 0.0
            outcome.responses += 1
            return outcome
        finally:
            st.close()

    def _resume(self, run_id: Any, units: Sequence[Unit], report: RunReport) -> None:
        """Finish earlier work before any new request."""
        for kind in ("judge", "generate"):
            for req in self.store.requests(kind=kind, status="submitted"):
                self.store.fail_request(
                    req["request_id"], status="api_error", error="interrupted before a response"
                )
        by_key = {u.key: u for u in units}
        for req in self.store.requests(kind="generate", status="received"):
            key = req["contrastive_group_id"] or str(req["combination_id"])
            unit = by_key.get(key)
            if unit is None:
                continue
            self.progress(f"resuming stored response for {unit.describe()}")
            report.add(self._process(self.store, run_id, req, unit))

    def _process(self, st: Store, run_id: Any, gen_req: Mapping[str, Any], unit: Unit) -> Outcome:
        outcome = Outcome()
        gen_id = gen_req["request_id"]
        parsed = parse_generation(
            self.cat,
            gen_req["response"]["text"],
            specs=unit.specs,
            count=gen_req["requested_count"],
            group=unit.group,
        )

        def reject(
            target_index, sample_index, stage, reason, gemini_label=None, judge=None, judge_id=None
        ):
            target = unit.targets[target_index if target_index is not None else 0]
            outcome.rejections[stage] += 1
            outcome.reasons[f"{stage}:{reason.split(' ')[0]}"] += 1
            return {
                "request_id": gen_id,
                "judge_request_id": judge_id,
                "combination_id": target["combination_id"],
                "language": unit.language,
                "run_id": run_id,
                "sample_index": sample_index,
                "stage": stage,
                "reason": reason,
                "gemini_label": gemini_label,
                "judge_label": judge,
            }

        if parsed.fatal:
            with st.transaction():
                st.insert_rejection(reject(None, None, "schema", parsed.fatal))
                st.fail_request(gen_id, status="schema_invalid", error=parsed.fatal)
            return outcome

        rejections = [
            reject(p.target_index, p.index, "schema", p.reason, p.label) for p in parsed.problems
        ]
        samples = parsed.samples
        judge_req = None
        verdicts: dict[int, Any] = {}
        if samples:
            judge_req = st.judge_request_for(gen_id)
            if judge_req is None:
                judge_req = self._judge(st, run_id, gen_id, unit, samples, outcome)
                if judge_req is None:
                    return outcome  # judge failed; the stored response is resumed later
            try:
                verdicts = parse_judgement(judge_req["response"]["text"], count=len(samples))
            except JudgementError as exc:
                st.fail_request(judge_req["request_id"], status="schema_invalid", error=str(exc))
                return outcome
        judge_id = judge_req["request_id"] if judge_req else None

        existing_hashes = st.text_hashes()
        batch_hashes: set[str] = set()
        decisions: list[tuple[Sample, Any, str | None, str | None]] = []
        for i, sample in enumerate(samples):
            verdict = verdicts[i]
            stage, reason = self._decide(unit.targets[sample.target_index], sample, verdict)
            if stage is None:
                h = text_hash(sample)
                if h in existing_hashes or h in batch_hashes:
                    stage, reason = "dedupe", "duplicate of an accepted conversation"
                batch_hashes.add(h)
            decisions.append((sample, verdict, stage, reason))

        if unit.group:
            failed_sets = {s.set_index for s, _, stage, _ in decisions if stage}
            decisions = [
                (s, v, "contrastive", "another member of the set was rejected")
                if stage is None and s.set_index in failed_sets
                else (s, v, stage, reason)
                for s, v, stage, reason in decisions
            ]

        conversations = []
        for sample, verdict, stage, reason in decisions:
            if stage:
                rejections.append(
                    reject(
                        sample.target_index,
                        sample.index,
                        stage,
                        reason,
                        sample.label,
                        verdict.label,
                        judge_id,
                    )
                )
                continue
            d = sample.data
            target = unit.targets[sample.target_index]
            conversations.append(
                {
                    "target": target,
                    "request_id": gen_id,
                    "judge_request_id": judge_id,
                    "run_id": run_id,
                    "sample_index": sample.index,
                    "turns": [
                        {
                            "speaker": d["context_turn"]["speaker"],
                            "role": d["context_turn"]["role"],
                            "text": d["context_turn"]["text"],
                        }
                    ],
                    "agent_speaker": d["agent_speaker"],
                    "agent_full_text": d["agent_full_text"],
                    "agent_spoken_text": d["agent_spoken_text"],
                    "candidate_speaker": d["candidate_speaker"],
                    "candidate_role": d["candidate_role"],
                    "candidate_text": d["candidate_text"],
                    "gemini_label": sample.label,
                    "gemini_rationale": d["rationale"],
                    "barge_in_from_word": sample.barge_in_from_word,
                    "judge_label": verdict.label,
                    "judge_barge_in_from_word": verdict.barge_in_from_word,
                    "judge_notes": verdict.notes,
                    "model_id": gen_req["model_id"],
                    "prompt_template": gen_req["prompt_template"],
                    "judge_model_id": judge_req["model_id"] if judge_req else None,
                    "text_hash": text_hash(sample),
                }
            )
        with st.transaction():
            for row in conversations:
                st.insert_conversation(row)
            for row in rejections:
                st.insert_rejection(row)
            st.mark_processed(gen_id)
            if judge_req is not None:
                st.mark_processed(judge_req["request_id"])
        outcome.accepted += len(conversations)
        return outcome

    def _judge(
        self,
        st: Store,
        run_id: Any,
        gen_id: Any,
        unit: Unit,
        samples: Sequence[Sample],
        outcome: Outcome,
    ) -> dict[str, Any] | None:
        items = [
            prompts.JudgeItem(
                index=i,
                spec=unit.specs[s.target_index],
                context_speaker=s.data["context_turn"]["speaker"],
                context_text=s.data["context_turn"]["text"],
                agent_speaker=s.data["agent_speaker"],
                agent_spoken_text=s.data["agent_spoken_text"],
                candidate_speaker=s.data["candidate_speaker"],
                candidate_text=s.data["candidate_text"],
            )
            for i, s in enumerate(samples)
        ]
        user = prompts.render_judge_request(self.cat, items)
        judge_id = st.insert_request(
            run_id=run_id,
            kind="judge",
            parent_request_id=gen_id,
            combination_id=None if unit.group else unit.targets[0]["combination_id"],
            contrastive_group_id=unit.key if unit.group else None,
            language=unit.language,
            requested_count=len(items),
            model_id=self.config.judge_model,
            prompt_template=prompts.JUDGE_TEMPLATE,
            prompt_hash=prompts.prompt_hash(self.judge_system, user),
        )
        try:
            result = self.client.generate(
                model=self.config.judge_model,
                system_prompt=self.judge_system,
                user_prompt=user,
                schema=prompts.JUDGE_SCHEMA,
                thinking_level=self.config.judge_thinking_level,
            )
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            st.fail_request(judge_id, status="api_error", error=redact(str(exc)))
            outcome.api_errors += 1
            return None
        cost = self.config.cost(result.usage)
        outcome.cost_usd += cost or 0.0
        st.record_response(judge_id, response=result.response, usage=result.usage, cost_usd=cost)
        return st.request(judge_id)

    def _decide(
        self, target: Mapping[str, Any], sample: Sample, verdict: Any
    ) -> tuple[str | None, str | None]:
        if verdict.checks_failed:
            notes = f" ({verdict.notes[:200]})" if verdict.notes else ""
            return "judge", ",".join(verdict.checks_failed) + notes
        intended = target["label"]
        gemini, judge = sample.label, verdict.label
        if intended == "ambiguous":
            if gemini != "ambiguous":
                return "label_mismatch", f"intended=ambiguous gemini={gemini} judge={judge}"
            return None, None
        if gemini != intended or judge != intended:
            return "label_mismatch", f"intended={intended} gemini={gemini} judge={judge}"
        if intended == "barge_in":
            g, j = sample.barge_in_from_word, verdict.barge_in_from_word
            if j is None or abs(g - j) > self.config.barge_in_word_tolerance:
                return "label_mismatch", f"barge_in_from_word gemini={g} judge={j}"
        return None, None
