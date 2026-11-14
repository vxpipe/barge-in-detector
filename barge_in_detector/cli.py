"""Command line: plan, generate, build examples, export and inspect.

Run from the repository root with `uv run python -m barge_in_detector <command>`.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import DEFAULT_PATH as DEFAULT_CONFIG
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.plan import PlanConfig, build_plan, split_audit
from barge_in_detector.storage.schema import SPEC_FACTORS
from barge_in_detector.storage.store import Store

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "work" / "barge_in.duckdb"
DEFAULT_EXPORT = REPO / "data" / "export"


def _table(rows: dict, title: str) -> None:
    print(title)
    for key, value in sorted(rows.items()):
        print(f"  {key:<40} {value}")


def cmd_check_catalogue(args: argparse.Namespace) -> int:
    cat = load_catalogue()
    errors = cat.check()
    print(f"catalogue {cat.version}: {cat.file_versions}")
    labels = Counter(i["label"] for i in cat.intents.values())
    print(
        f"{len(cat.languages)} languages, {len(cat.domains)} domains, {len(cat.scenarios)} scenarios, "
        f"{len(cat.intents)} intents {dict(labels)}, {len(cat.rules)} rules, "
        f"{len(cat.contrastive)} contrastive groups"
    )
    for error in errors:
        print(f"ERROR {error}")
    print("no errors" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


def cmd_plan(args: argparse.Namespace) -> int:
    cat = load_catalogue()
    config = PlanConfig.from_file(args.plan)
    store = Store.open(args.db)
    try:
        plan = build_plan(cat, config, existing=store.existing_targets())
        audit = split_audit(
            cat,
            plan.targets,
            config.split_ratios,
            tolerance=config.split_tolerance,
            min_support=config.split_min_support,
        )
        by = Counter(
            (t.spec["language"], t.intended_label, t.quota_group, t.split) for t in plan.targets
        )
        _table(
            {" ".join(k): v for k, v in by.items()},
            f"plan {config.name}: {len(plan.targets)} criteria",
        )
        samples = sum(t.required_count for t in plan.targets)
        print(f"required samples: {samples}")
        for failure in audit.failures:
            print(f"SPLIT {failure}")
        if audit.failures:
            print("split audit failed; plan not applied")
            return 1
        if args.dry_run:
            print("dry run: nothing written")
            return 0
        revision_id = store.apply_plan(cat, plan)
        if revision_id is None:
            print("plan unchanged: no new criteria or counts")
            return 0
        print(f"applied as plan revision {revision_id}")
        changed = store.export(args.export_dir)
        print(f"exported: {len(changed)} files changed under {args.export_dir}")
        return 0
    finally:
        store.close()


def _client(config: GenerationConfig):
    from barge_in_detector.generation.llm import GeminiClient

    return GeminiClient(
        temperature=config.temperature,
        explicit_cache=config.explicit_cache,
        cache_ttl_seconds=config.cache_ttl_seconds,
    )


class _NoClient:
    def generate(self, **kwargs):  # pragma: no cover - dry runs never call it
        raise RuntimeError("dry run")

    def close(self) -> None:
        pass


def cmd_generate(args: argparse.Namespace) -> int:
    from barge_in_detector.generation.runner import Runner

    cat = load_catalogue()
    config = GenerationConfig.from_file(args.config)
    if args.concurrency:
        config = GenerationConfig(**{**config.__dict__, "concurrency": args.concurrency})
    languages = args.languages.split(",") if args.languages else None
    store = Store.open(args.db)
    try:
        client = _NoClient() if args.dry_run else _client(config)
        runner = Runner(store, cat, client, config, progress=print)
        report = runner.run(
            languages=languages,
            limit=args.limit,
            budget_usd=args.budget_usd,
            dry_run=args.dry_run,
            run_name=args.name,
            export_dir=None if args.no_export or args.dry_run else args.export_dir,
        )
    finally:
        store.close()
    if args.dry_run:
        print(
            f"would make {report.planned_requests} generator requests for {report.planned_samples} samples"
        )
        _table(dict(report.planned_by_label), "samples needed by language:label")
        print(f"estimated tokens: {report.estimated_tokens}")
        cost = report.estimated_cost_usd
        print("estimated cost: " + (f"${cost:.2f}" if cost is not None else "unknown (no prices)"))
    else:
        print(
            f"run {report.run_id}: {report.requests} requests, {report.accepted} accepted, "
            f"{sum(report.rejections.values())} rejected, {report.api_errors} API errors, "
            + (f"cost ${report.cost_usd:.4f}" if report.cost_usd is not None else "cost unknown")
        )
        if report.reasons:
            _table(dict(report.reasons), "rejections by stage:reason")
        print(f"criteria still short: {len(report.short)}")
    if report.exhausted:
        print(f"attempt budget exhausted for {len(report.exhausted)} criteria/groups:")
        for item in report.exhausted[:20]:
            print(f"  {item}")
    return 0


def cmd_build_examples(args: argparse.Namespace) -> int:
    from barge_in_detector.examples import BUILDER_VERSION, build_examples

    store = Store.open(args.db)
    try:
        rows = build_examples(store.conversations())
        store.replace_examples(BUILDER_VERSION, rows)
        print(f"builder {BUILDER_VERSION}: {len(rows)} examples")
        if not args.no_export:
            store.export(args.export_dir)
    finally:
        store.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store.open(args.db)
    try:
        changed = store.export(args.export_dir)
    finally:
        store.close()
    print(f"{len(changed)} files changed under {args.export_dir}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    if Path(args.db).exists():
        print(f"{args.db} exists; move it away to rebuild from exports", file=sys.stderr)
        return 1
    store = Store.open(args.db)
    try:
        store.import_exports(args.export_dir)
        counts = {t: store.count(t) for t in ("targets", "requests", "conversations", "examples")}
    finally:
        store.close()
    print(f"imported {counts}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = Store.open(args.db)
    try:
        accepted = store.accepted_counts()
        rows: Counter = Counter()
        for t in store.targets():
            key = f"{t['language']} {t['label']}"
            rows[f"{key} required"] += t["required_count"]
            rows[f"{key} accepted"] += min(
                accepted.get(t["combination_id"], 0), t["required_count"]
            )
        _table(dict(rows), "samples")
        stages = store.con.execute(
            "SELECT stage, count(*) FROM rejections GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        _table(dict(stages), "rejections by stage")
    finally:
        store.close()
    return 0


def cmd_show_prompt(args: argparse.Namespace) -> int:
    cat = load_catalogue()
    if args.system:
        print("=" * 30 + " generator system prompt (cached)")
        print(prompts.render_generator_system(cat))
        print("=" * 30 + " judge system prompt (cached)")
        print(prompts.render_judge_system(cat))
        return 0
    store = Store.open(args.db)
    try:
        targets = store.targets()
    finally:
        store.close()
    if not targets:
        print("no targets; run the plan command first", file=sys.stderr)
        return 1
    target = targets[args.index % len(targets)]
    group_id = target["contrastive_group_id"]
    if group_id:
        members = sorted(
            (t for t in targets if t["contrastive_group_id"] == group_id),
            key=lambda t: t["contrastive_role"],
        )
        group = next(g for g in cat.contrastive if g["id"] == group_id.split(":", 1)[0])
        specs = [{f: t[f] for f in SPEC_FACTORS} for t in members]
        sets = max(1, args.count // len(members))  # as the runner sizes a request
        print(prompts.render_generation_request(cat, specs, count=sets, group=group))
    else:
        spec = {f: target[f] for f in SPEC_FACTORS}
        print(prompts.render_generation_request(cat, [spec], count=args.count))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m barge_in_detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_db(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--db", type=Path, default=DEFAULT_DB, help="working DuckDB file")
        p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
        return p

    sub.add_parser("check-catalogue", help="validate generation/factors").set_defaults(
        func=cmd_check_catalogue
    )

    p = with_db(sub.add_parser("plan", help="build a plan revision and apply it"))
    p.add_argument("plan", type=Path, help="plan file, e.g. generation/plans/en-hi-pilot.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = with_db(sub.add_parser("generate", help="fill every criterion to its required count"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--languages", help="comma-separated language codes")
    p.add_argument("--limit", type=int, help="maximum generator requests in this run")
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--name", default="generate")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = with_db(sub.add_parser("build-examples", help="rebuild examples from conversations"))
    p.add_argument("--no-export", action="store_true")
    p.set_defaults(func=cmd_build_examples)

    with_db(sub.add_parser("export", help="write data/export")).set_defaults(func=cmd_export)
    with_db(sub.add_parser("import", help="rebuild the database from data/export")).set_defaults(
        func=cmd_import
    )
    with_db(sub.add_parser("status", help="filled and short counts")).set_defaults(func=cmd_status)

    p = with_db(sub.add_parser("show-prompt", help="print the prompts for review"))
    p.add_argument("--system", action="store_true", help="print the cached system prompts")
    p.add_argument("--index", type=int, default=0, help="which target (plan order)")
    p.add_argument("--count", type=int, default=10, help="samples per request (batch size)")
    p.set_defaults(func=cmd_show_prompt)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
