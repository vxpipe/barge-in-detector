"""Judge-free supplement generation, with durable request-level cost reservations.

Use a separate working store. Every submitted request reserves the fixed maximum
in the versioned plan until complete token usage arrives. Lost/failed responses
retain that reservation across restarts. The SDK and wrapper make no retries;
each runner retry is another logged, budgeted request. No explicit cache costs.
"""

import argparse
import hashlib
import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, date, datetime
from pathlib import Path

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.cleaner import clean_text
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.llm import GeminiClient, redact
from barge_in_detector.generation.phrase_coverage import (
    KIND,
    PLAN_PATH,
    SCHEMA,
    TEMPLATE,
    build_targets,
    coverage_spec,
    load_plan,
    render,
    sample_problem,
)
from barge_in_detector.generation.prompts import prompt_hash
from barge_in_detector.splitting import read_csv
from barge_in_detector.storage.store import Store


def request_reserve(plan):
    prices = plan["prices_usd_per_million"]
    return (
        plan["max_input_tokens"] * prices["input"] + plan["max_output_tokens"] * prices["output"]
    ) / 1e6


def input_bound(system, user):
    # Text-only request. UTF-8 bytes conservatively overcount text tokens;
    # include the schema and a generous allowance for message framing.
    return (
        len(
            json.dumps(
                {"system": system, "user": user, "schema": SCHEMA}, ensure_ascii=False
            ).encode()
        )
        + 1024
    )


def usage_cost(plan, usage):
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if any(not isinstance(v, int) or v < 0 for v in (input_tokens, output_tokens)):
        return None
    prices = plan["prices_usd_per_million"]
    # Charge cached input at full price: the budget never relies on a cache hit.
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1e6


class PhraseRunner:
    def __init__(self, store, cat, plan, targets, client, *, excluded, model=None):
        self.store, self.cat, self.plan, self.client = store, cat, plan, client
        self.targets = {t["combination_id"]: t for t in targets}
        self.excluded = dict(excluded)
        self.model = model or GenerationConfig.from_file().generator_model
        if self.model != plan["priced_model"]:
            raise ValueError("Configured generator does not match the plan's verified prices")
        bounds = {
            k: plan[k]
            for k in (
                "max_input_tokens",
                "max_output_tokens",
                "prices_usd_per_million",
                "thinking_level",
            )
        }
        signature = hashlib.sha256(json.dumps(bounds, sort_keys=True).encode()).hexdigest()[:12]
        self.template = f"{TEMPLATE}:{signature}"
        compatible = [f"{t}:{signature}" for t in ("generator_phrases_v1", TEMPLATE)]
        incompatible = store.con.execute(
            "SELECT count(*) FROM requests WHERE kind != ? OR model_id != ? "
            "OR prompt_template NOT IN (?, ?)",
            [KIND, self.model, *compatible],
        ).fetchone()[0]
        if incompatible:
            raise ValueError(
                "Use the original phrase plan/model/bounds for this store; preserve its ledger"
            )

    def committed_cost(self):
        known, unknown = self.store.con.execute(
            "SELECT coalesce(sum(cost_usd), 0), count(*) FILTER (WHERE cost_usd IS NULL) FROM requests"
        ).fetchone()
        return known + unknown * request_reserve(self.plan)

    def submit_record(self, target, count):
        system, user = render(self.cat, target, count)
        if input_bound(system, user) > self.plan["max_input_tokens"]:
            raise ValueError("Prompt exceeds the reserved input allowance")
        if self.committed_cost() + request_reserve(self.plan) > self.plan["new_budget_usd"]:
            raise ValueError("Budget exhausted before submission")
        return self.store.insert_request(
            kind=KIND,
            combination_id=target["combination_id"],
            language=target["language"],
            requested_count=count,
            model_id=self.model,
            prompt_template=self.template,
            prompt_hash=prompt_hash(system, user),
        )

    def _process(self, req):
        target = self.targets[req["combination_id"]]
        try:
            samples = json.loads(req["response"]["text"])["samples"]
            if not isinstance(samples, list):
                raise ValueError("samples is not a list")
        except (KeyError, TypeError, ValueError) as exc:
            self.store.fail_request(req["request_id"], status="schema_invalid", error=str(exc))
            return
        existing = {r["text_clean"] for r in self.store.utterances()}
        with self.store.transaction():
            for index, sample in enumerate(samples):
                problem = sample_problem(self.cat, target, sample)
                text = sample.get("text") if isinstance(sample, dict) else None
                cleaned = clean_text(text) if isinstance(text, str) else None
                if len(samples) != req["requested_count"]:
                    problem = "count_mismatch"
                if problem is None:
                    problem = self.excluded.get(cleaned)
                if problem is None and cleaned in existing:
                    problem = "duplicate_supplement"
                self.store.insert_utterance(
                    target,
                    {
                        "status": "rejected" if problem else "accepted",
                        "reason": problem,
                        "text": text,
                        "label": sample.get("label") if isinstance(sample, dict) else None,
                        "raw_sample": {
                            "sample": sample,
                            "coverage": coverage_spec(target),
                            "verification": "local_checks_only_no_judge",
                        },
                        "text_hash": hashlib.sha256(cleaned.encode()).hexdigest()
                        if cleaned
                        else None,
                        "request_id": req["request_id"],
                        "sample_index": index,
                        "model_id": req["model_id"],
                        "prompt_template": req["prompt_template"],
                    },
                )
                if problem is None:
                    existing.add(cleaned)
            self.store.mark_processed(req["request_id"])

    def resume(self):
        # A received response is reusable. A lost in-flight response remains
        # charged at its maximum; don't send that request invisibly again.
        for req in self.store.requests(kind=KIND, status="submitted"):
            self.store.fail_request(
                req["request_id"], status="api_error", error="Interrupted; maximum cost retained"
            )
        for req in self.store.requests(kind=KIND, status="received"):
            if req["combination_id"] not in self.targets:
                raise ValueError("Plan removed a criterion with a response awaiting resume")
            self._process(req)

    def requested_counts(self):
        return Counter(
            dict(
                self.store.con.execute(
                    "SELECT combination_id, sum(requested_count) FROM requests GROUP BY 1"
                ).fetchall()
            )
        )

    def needs_generation(self):
        if self.committed_cost() + request_reserve(self.plan) > self.plan["new_budget_usd"]:
            return False
        accepted, attempted = self.store.accepted_counts(), self.requested_counts()
        return any(
            accepted.get(cid, 0) < t["required_count"]
            and attempted[cid] < self.plan["max_attempts"] * t["required_count"]
            for cid, t in self.targets.items()
        )

    def run(self, export_dir, *, limit_requests=None):
        self.resume()
        attempts = self.requested_counts()
        accepted = Counter(self.store.accepted_counts())
        queue = deque(self.targets)
        submitted = completed = 0
        abort_reason = None
        try:
            with ThreadPoolExecutor(max_workers=self.plan["concurrency"]) as pool:
                pending = {}
                while queue or pending:
                    while queue and len(pending) < self.plan["concurrency"]:
                        if limit_requests is not None and submitted >= limit_requests:
                            queue.clear()
                            break
                        if (
                            self.committed_cost() + request_reserve(self.plan)
                            > self.plan["new_budget_usd"]
                        ):
                            # In-flight requests can release unused reservations.
                            if not pending:
                                queue.clear()
                            break
                        cid = queue.popleft()
                        target = self.targets[cid]
                        need = target["required_count"] - accepted[cid]
                        allowance = (
                            self.plan["max_attempts"] * target["required_count"] - attempts[cid]
                        )
                        if need <= 0 or allowance <= 0:
                            continue
                        need = min(need, allowance)
                        rid = self.submit_record(target, need)
                        system, user = render(self.cat, target, need)
                        future = pool.submit(
                            self.client.generate,
                            model=self.model,
                            system_prompt=system,
                            user_prompt=user,
                            schema=SCHEMA,
                            thinking_level=self.plan["thinking_level"],
                            max_output_tokens=self.plan["max_output_tokens"],
                        )
                        pending[future] = (rid, cid)
                        submitted += 1
                        attempts[cid] += need
                    if not pending:
                        continue
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        rid, cid = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - record failed call; retain reservation
                            message = redact(str(exc))
                            self.store.fail_request(rid, status="api_error", error=message)
                            if message.startswith(("400:", "401:", "403:", "404:")):
                                abort_reason = message
                                queue.clear()
                        else:
                            cost = usage_cost(self.plan, result.usage)
                            self.store.record_response(
                                rid, response=result.response, usage=result.usage, cost_usd=cost
                            )
                            if (result.usage.get("input_tokens") or 0) > self.plan[
                                "max_input_tokens"
                            ] or (result.usage.get("output_tokens") or 0) > self.plan[
                                "max_output_tokens"
                            ]:
                                raise RuntimeError(
                                    "Provider exceeded token reservation; stop and review pricing"
                                )
                            self._process(self.store.request(rid))
                        accepted = Counter(self.store.accepted_counts())
                        if abort_reason is None:
                            queue.append(cid)
                        completed += 1
                        if completed % 25 == 0:
                            self.store.export(export_dir)
                            print(
                                json.dumps(
                                    {
                                        "completed_this_run": completed,
                                        "accepted": sum(accepted.values()),
                                        "cost_plus_reservations_usd": round(
                                            self.committed_cost(), 4
                                        ),
                                    }
                                ),
                                flush=True,
                            )
        finally:
            self.store.export(export_dir)
        return {
            "submitted_this_run": submitted,
            "accepted": sum(accepted.values()),
            "recorded_cost_usd": self.store.total_cost(),
            "cost_plus_reservations_usd": self.committed_cost(),
            "short_criteria": sum(
                accepted[cid] < t["required_count"] for cid, t in self.targets.items()
            ),
            "abort_reason": abort_reason,
        }


def estimate(cat, plan, targets):
    if GenerationConfig.from_file().generator_model != plan["priced_model"]:
        raise ValueError("Configured generator does not match the plan's verified prices")
    prompts = [render(cat, t, t["required_count"]) for t in targets]
    if max(input_bound(s, u) for s, u in prompts) > plan["max_input_tokens"]:
        raise ValueError("A planned prompt exceeds input reservation")
    expected_input = sum((input_bound(s, u) - 1024) / 3 for s, u in prompts)
    expected_output = len(targets) * (plan["required_count"] * 120 + 256)
    prices = plan["prices_usd_per_million"]
    expected = (expected_input * prices["input"] + expected_output * prices["output"]) / 1e6
    return {
        "model": GenerationConfig.from_file().generator_model,
        "judge_requests": 0,
        "languages": len(plan["languages"]),
        "criteria": len(targets),
        "requested_unique_samples": sum(t["required_count"] for t in targets),
        "expected_first_pass_usd": round(expected, 4),
        "expected_with_full_second_pass_usd": round(expected * plan["max_attempts"], 4),
        "assumptions": "Input UTF-8 bytes/3; 120 output tokens/sample plus 256 thinking/request. Estimates, not quotes.",
        "maximum_per_request_usd": request_reserve(plan),
        "full_attempts_token_maximum_before_budget_stop_usd": round(
            len(targets) * plan["max_attempts"] * plan["required_count"] * request_reserve(plan), 4
        ),
        "enforced_new_run_cap_usd": plan["new_budget_usd"],
        "historical_recorded_usd": plan["historical_recorded_usd"],
        "historical_uncertainty_reserve_usd": plan["historical_uncertainty_reserve_usd"],
        "maximum_accounted_total_usd": plan["historical_recorded_usd"]
        + plan["historical_uncertainty_reserve_usd"]
        + plan["new_budget_usd"],
        "total_ceiling_usd": plan["total_ceiling_usd"],
        "prices_source": "https://aistudio.google.com/docs/pricing#gemini-3.8-flash",
    }


def exclusions():
    result = {
        r["text_clean"]: "duplicate_original"
        for r in read_csv(Path("data/export/utterances.csv"))
        if r["status"] == "accepted"
    }
    for path in Path("data/evaluation").glob("*.csv"):
        for row in read_csv(path):
            result[clean_text(row.get("text_clean") or row["text"])] = "reserved_evaluation"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument(
        "--run", action="store_true", help="Make paid calls; default is an offline estimate"
    )
    parser.add_argument("--limit-requests", type=int)
    args = parser.parse_args()
    cat, plan = load_catalogue(), load_plan(args.plan)
    targets = build_targets(cat, plan)
    report_dir = Path("reports/phrase-coverage-v1")
    report_dir.mkdir(parents=True, exist_ok=True)
    estimated = estimate(cat, plan, targets)
    estimate_path = report_dir / "cost-estimate.json"
    estimate_path.write_text(json.dumps(estimated, indent=2) + "\n")
    print(json.dumps(estimated, indent=2), flush=True)
    if not args.run:
        return
    store = Store.open(Path("data/work/phrase-coverage-v1.duckdb"))
    config = GenerationConfig.from_file()
    client = None
    try:
        runner = PhraseRunner(store, cat, plan, targets, None, excluded=exclusions())
        runner.resume()
        if runner.needs_generation() and args.limit_requests != 0:
            if datetime.now(UTC).date() > date.fromisoformat(plan["prices_valid_through"]):
                raise ValueError("Pricing expired; review before generation")
            client = GeminiClient(
                temperature=config.temperature, explicit_cache=False, max_retries=0
            )
            client.check_model_access(config.generator_model)
            runner.client = client
        result = runner.run(
            Path("data/supplements/phrase-coverage-v1"), limit_requests=args.limit_requests
        )
        (report_dir / "generation-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if client is not None:
            client.close()
        store.close()


if __name__ == "__main__":
    main()
