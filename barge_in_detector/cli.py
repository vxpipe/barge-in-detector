"""Command line: plan, generate, status, export and inspect.

Run from the repository root with `uv run python -m barge_in_detector <command>`.
The plan file (generation/plans/) is the only record of the criteria.
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
from barge_in_detector.generation.utterances import (
    UtterancePlanConfig,
    build_utterance_plan,
    load_utterance_track,
    plan_targets,
)
from barge_in_detector.storage.store import Store

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "work" / "barge_in.duckdb"
DEFAULT_EXPORT = REPO / "data" / "export"
DEFAULT_PLAN = REPO / "generation" / "plans" / "utterance-sample.yaml"


def _table(rows: dict, title: str) -> None:
    print(title)
    for key, value in sorted(rows.items()):
        print(f"  {key:<40} {value}")


def _targets(plan_path: Path) -> tuple:
    cat = load_catalogue()
    track = load_utterance_track(cat)
    config = UtterancePlanConfig.from_file(plan_path)
    return cat, track, config, plan_targets(cat, build_utterance_plan(cat, track, config))


def cmd_check_catalogue(args: argparse.Namespace) -> int:
    cat = load_catalogue()
    errors = cat.check()
    print(f"catalogue {cat.version}: {cat.file_versions}")
    labels = Counter(i["label"] for i in cat.intents.values())
    print(f"{len(cat.languages)} languages, {len(cat.intents)} intents {dict(labels)}")
    for error in errors:
        print(f"ERROR {error}")
    print("no errors" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


def cmd_plan(args: argparse.Namespace) -> int:
    _, _, config, targets = _targets(args.plan)
    by = Counter((t["language"], t["label"]) for t in targets)
    _table({" ".join(k): v for k, v in by.items()}, f"plan {config.name}: {len(targets)} criteria")
    print(f"required utterances: {sum(t['required_count'] for t in targets)}")
    return 0


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
    from barge_in_detector.generation.utterance_runner import UtteranceRunner

    cat, track, _, targets = _targets(args.plan)
    config = GenerationConfig.from_file(args.config)
    if args.concurrency:
        config = GenerationConfig(**{**config.__dict__, "concurrency": args.concurrency})
    languages = args.languages.split(",") if args.languages else None
    store = Store.open(args.db)
    try:
        client = _NoClient() if args.dry_run else _client(config)
        runner = UtteranceRunner(store, cat, track, client, config, targets, progress=print)
        report = runner.run(
            languages=languages,
            limit=args.limit,
            budget_usd=args.budget_usd,
            dry_run=args.dry_run,
            export_dir=None if args.no_export or args.dry_run else args.export_dir,
        )
    finally:
        store.close()
    if args.dry_run:
        print(
            f"would make {report.planned_requests} generator requests for "
            f"{report.planned_samples} utterances"
        )
        _table(dict(report.planned_by_label), "utterances needed by language:label")
        print(f"estimated tokens: {report.estimated_tokens}")
        cost = report.estimated_cost_usd
        print("estimated cost: " + (f"${cost:.4f}" if cost is not None else "unknown (no prices)"))
    else:
        print(
            f"{report.requests} generator requests, {report.judge_requests} judge requests, "
            f"{report.accepted} accepted, {sum(report.rejections.values())} rejected, "
            f"{report.api_errors} API errors, "
            + (f"cost ${report.cost_usd:.4f}" if report.cost_usd is not None else "cost unknown")
        )
        if report.reasons:
            _table(dict(report.reasons), "rejections by stage:reason")
        print(f"criteria still short: {len(report.short)}")
    if report.exhausted:
        print(f"attempt budget exhausted for {len(report.exhausted)} criteria:")
        for item in report.exhausted[:20]:
            print(f"  {item}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from barge_in_detector.splitting import read_csv, split_rows, write_splits

    splits = split_rows(read_csv(args.export_dir / "utterances.csv"))
    write_splits(splits, args.out)
    for name, items in splits.items():
        labels = Counter(r["label"] for r in items)
        print(f"{name}: {len(items)} ({dict(sorted(labels.items()))}) -> {args.out / name}.csv")
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
        count = store.count("utterances")
    finally:
        store.close()
    print(f"imported {count} utterances")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _, _, config, targets = _targets(args.plan)
    store = Store.open(args.db)
    try:
        accepted = store.accepted_counts()
        rows: Counter = Counter()
        for t in targets:
            key = f"{t['language']} {t['label']}"
            rows[f"{key} required"] += t["required_count"]
            rows[f"{key} accepted"] += min(
                accepted.get(t["combination_id"], 0), t["required_count"]
            )
        _table(dict(rows), f"plan {config.name}")
        statuses = store.con.execute(
            "SELECT status, count(*) FROM utterances GROUP BY 1 ORDER BY 1"
        ).fetchall()
        _table(dict(statuses), "samples by status")
        print(f"cost so far (local request log): ${store.total_cost():.4f}")
    finally:
        store.close()
    return 0


def cmd_show_prompt(args: argparse.Namespace) -> int:
    cat, track, _, targets = _targets(args.plan)
    if args.system:
        print(prompts.render_utterance_system(cat, track))
        return 0
    target = targets[args.index % len(targets)]
    spec = {f: target[f] for f in track.factors}
    print(prompts.render_utterance_request(cat, spec, count=args.count))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m barge_in_detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_plan(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="plan file")
        return p

    def with_db(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--db", type=Path, default=DEFAULT_DB, help="working DuckDB file")
        p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
        return p

    sub.add_parser("check-catalogue", help="validate generation/factors").set_defaults(
        func=cmd_check_catalogue
    )
    with_plan(sub.add_parser("plan", help="show the criteria a plan file builds")).set_defaults(
        func=cmd_plan
    )

    p = with_plan(with_db(sub.add_parser("generate", help="fill every criterion to its count")))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--languages", help="comma-separated language codes")
    p.add_argument("--limit", type=int, help="maximum generator requests in this run")
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--concurrency", type=int, help="parallel Gemini calls (config: concurrency)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = with_db(sub.add_parser("split", help="write train/validation/test CSVs for training"))
    p.add_argument("--out", type=Path, default=REPO / "data" / "splits")
    p.set_defaults(func=cmd_split)

    with_db(sub.add_parser("export", help="write data/export/utterances.csv")).set_defaults(
        func=cmd_export
    )
    with_db(sub.add_parser("import", help="rebuild the database from the export")).set_defaults(
        func=cmd_import
    )
    with_plan(with_db(sub.add_parser("status", help="filled and short counts"))).set_defaults(
        func=cmd_status
    )

    p = with_plan(sub.add_parser("show-prompt", help="print the generator prompts for review"))
    p.add_argument("--system", action="store_true", help="the cached system prompt")
    p.add_argument("--index", type=int, default=0, help="which criterion (plan order)")
    p.add_argument("--count", type=int, default=5, help="samples per request")
    p.set_defaults(func=cmd_show_prompt)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
