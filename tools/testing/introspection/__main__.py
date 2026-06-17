"""testintro CLI.

Run via the launcher (auto-uses the repo .venv):

    tools/testing/testintro platforms | configs
    tools/testing/testintro list  --job <platform/config> [--file] [--status] [--count] [--files] [--json]
    tools/testing/testintro diff  --job <platform/config> (--pr N | --from A [--to B]) [--variants] [--json]
    tools/testing/testintro where <query> [--file] [--platforms] [--config] [--json]

(equivalently `python -m tools.testing.introspection <...>`.)

Behaviors:
  list  (B3) -- concrete tests a CI job runs, aggregated over the config's files
  diff  (B1) -- tests added/removed between two git refs
  where (B2) -- which platforms a named test runs on (RUNS/SKIPPED/NOT_GENERATED)
"""

from __future__ import annotations

import argparse
import json
import sys

from tools.testing.introspection import (
    collector,
    diff as diff_mod,
    platforms,
    where as where_mod,
)


def _cmd_platforms(args: argparse.Namespace) -> int:
    for name in sorted(platforms.REGISTRY):
        p = platforms.REGISTRY[name]
        cap = f" sm{p.cuda_capability}" if p.cuda_capability else ""
        print(f"{name:18} device={p.device_type}{' rocm' if p.rocm else ''}{cap}")
    return 0


def _cmd_configs(args: argparse.Namespace) -> int:
    for name in sorted(platforms.CONFIGS):
        c = platforms.CONFIGS[name]
        print(f"{name:14} options={c.options or '{}'} env={c.env or '{}'}")
    return 0


def _list_one_file(args: argparse.Namespace, job: platforms.Job) -> int:
    use_cache = not args.no_cache
    if args.status:
        st = collector.status(args.file, job, use_cache=use_cache)
        skipped = dict(st["skipped"])
        if args.json:
            print(json.dumps({"job": job.name, "file": args.file, **st}, indent=2))
            return 0
        print(
            f"# {args.file} on {job.name}: {len(st['ran'])} run, {len(skipped)} skipped"
        )
        for t in st["ran"]:
            print(f"RUNS     {t}")
        for t, reason in sorted(skipped.items()):
            print(f"SKIPPED  {t}    [{reason}]")
        return 0
    classes = collector.enumerate_tests(args.file, job, use_cache=use_cache)
    total = sum(len(v) for v in classes.values())
    if args.json:
        print(
            json.dumps(
                {
                    "job": job.name,
                    "file": args.file,
                    "total": total,
                    "classes": classes,
                },
                indent=2,
            )
        )
        return 0
    print(f"# {args.file} on {job.name}: {total} tests")
    for cls in sorted(classes):
        for m in classes[cls]:
            print(f"{args.file}::{cls}::{m}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    job = platforms.get_job(args.job)
    if args.file:
        return _list_one_file(args, job)

    agg = collector.list_job(
        job,
        with_status=args.status,
        files_filter=args.files,
        use_cache=not args.no_cache,
    )
    results = agg["results"]
    errors = {f: r["error"] for f, r in results.items() if "error" in r}

    if args.json:
        print(json.dumps(agg, indent=2))
        return 0

    if args.status:
        ran = skipped = 0
        per_file = []
        for f, r in results.items():
            if "error" in r:
                continue
            nr = len(r["status"]["ran"])
            ns = len(r["status"]["skipped"])
            ran += nr
            skipped += ns
            per_file.append((f, nr, ns))
        print(
            f"# {job.name}: {len(agg['files'])} files, {ran} run, {skipped} skipped, "
            f"{len(errors)} errors, {len(agg['missing'])} non-py"
        )
        if not args.count:
            for f, nr, ns in sorted(per_file):
                print(f"{nr:6} run {ns:5} skip   {f}")
    else:
        total = 0
        per_file = []
        for f, r in results.items():
            if "error" in r:
                continue
            n = sum(len(v) for v in r["classes"].values())
            total += n
            per_file.append((f, n))
        print(
            f"# {job.name}: {len(agg['files'])} files, {total} tests, "
            f"{len(errors)} errors, {len(agg['missing'])} non-py"
        )
        if not args.count:
            for f, r in sorted(results.items()):
                if "error" in r:
                    continue
                for cls in sorted(r["classes"]):
                    for m in r["classes"][cls]:
                        print(f"{f}::{cls}::{m}")

    for f, msg in sorted(errors.items()):
        print(f"ERROR  {f}: {msg}", file=sys.stderr)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    if args.job:
        jobs = [platforms.get_job(j.strip()) for j in args.job.split(",")]
    else:
        jobs = [platforms.get_job(p) for p in sorted(platforms.REGISTRY)]
    if args.pr:
        a, b = diff_mod.resolve_pr(args.pr)
    else:
        a, b = args.__dict__["from"], args.to
        if a is None:
            print("error: pass --pr N or --from REF", file=sys.stderr)
            return 2
    res = diff_mod.diff(
        jobs, a, b, files_filter=args.files, full=args.full, locations=args.locations
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    # Invert per-job results to per-test: which platforms added / removed each test.
    added, removed, job_names = diff_mod.invert_per_job(res)
    all_jobs = set(job_names)

    def _plats(s: set[str]) -> str:
        return "all platforms" if s == all_jobs else ", ".join(sorted(s))

    print(f"# {res['from']}..{res['to']}  ({len(job_names)} platforms)")
    print(f"# +{len(added)} added, -{len(removed)} removed (distinct tests)")
    cap = None if args.variants else 40

    def _section(title: str, sign: str, m: dict[str, set[str]]) -> None:
        if not m:
            return
        print(f"\n{title}:")
        for fs, tests in diff_mod.group_by_platform_set(m, all_jobs):
            for t in tests[:cap]:
                print(f"{sign} {t}")
            if cap and len(tests) > cap:
                print(f"{sign} ... {len(tests) - cap} more (use --variants)")
            print(f"      {_plats(set(fs))}")

    _section("ADDED", "+", added)
    _section("REMOVED", "-", removed)

    unc = sorted(
        {f for jr in res["per_job"].values() for f in jr.get("uncomparable", [])}
    )
    if unc:
        print(
            f"\n# {len(unc)} files uncomparable (collect error at a ref): {', '.join(unc[:5])}"
            + (" ..." if len(unc) > 5 else "")
        )
    return 0


def _cmd_where(args: argparse.Namespace) -> int:
    plats = args.platforms.split(",") if args.platforms else None
    res = where_mod.where(
        args.query, file=args.file, platform_names=plats, config=args.config
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    if not res["files"]:
        print(f"# no test matching {args.query!r} found")
        return 0
    print(f"# where {args.query!r} (config={args.config})")
    for f, per_plat in res["files"].items():
        print(f"\n{f}")
        for plat in res["platforms"]:
            r = per_plat.get(plat, {})
            if "error" in r:
                print(f"  {plat:18} ERROR  {r['error'][:60]}")
            elif r["runs"]:
                extra = f"  (+{len(r['skipped'])} skipped)" if r["skipped"] else ""
                print(
                    f"  {plat:18} RUNS({len(r['runs'])}){extra}  {', '.join(r['runs'][:3])}"
                )
            elif r["skipped"]:
                reason = r["skipped"][0][1]
                print(f"  {plat:18} SKIPPED({len(r['skipped'])})  [{reason[:50]}]")
            else:
                print(f"  {plat:18} NOT_GENERATED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="testintro")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("platforms", help="list known device platforms").set_defaults(
        func=_cmd_platforms
    )
    sub.add_parser("configs", help="list known CI configs").set_defaults(
        func=_cmd_configs
    )

    p_list = sub.add_parser("list", help="list tests a job runs (behavior 3)")
    p_list.add_argument(
        "--job", required=True, help="platform/config, e.g. linux-cuda-sm80/default"
    )
    p_list.add_argument(
        "--file", help="scope to a single repo-relative file (fast path)"
    )
    p_list.add_argument("--status", action="store_true", help="split RUNS vs SKIPPED")
    p_list.add_argument(
        "--count", action="store_true", help="summary only, no per-test listing"
    )
    p_list.add_argument("--files", help="only files whose path contains this substring")
    p_list.add_argument(
        "--no-cache", action="store_true", help="ignore the content-hash cache"
    )
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_diff = sub.add_parser(
        "diff", help="tests added/removed between two refs (behavior 1)"
    )
    p_diff.add_argument(
        "--job",
        help="comma-separated platform/config (default: all platforms), "
        "e.g. linux-cuda-sm80/default,linux-cpu/default",
    )
    p_diff.add_argument("--pr", type=int, help="GitHub PR number (fetches base+head)")
    p_diff.add_argument("--from", help="base ref (when not using --pr)")
    p_diff.add_argument("--to", default="HEAD", help="head ref (default HEAD)")
    p_diff.add_argument("--files", help="only files whose path contains this substring")
    p_diff.add_argument(
        "--full", action="store_true", help="skip scoping; diff the whole selected set"
    )
    p_diff.add_argument(
        "--variants",
        action="store_true",
        help="show all added/removed, not just first 10",
    )
    p_diff.add_argument(
        "--locations",
        action="store_true",
        help="also report each test's source file/line (added_loc/removed_loc in --json)",
    )
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=_cmd_diff)

    p_where = sub.add_parser(
        "where", help="which platforms a test runs on (behavior 2)"
    )
    p_where.add_argument(
        "query", help="test id (Class::method), bare method, or substring"
    )
    p_where.add_argument(
        "--file", help="repo-relative file (recommended; else scans the universe)"
    )
    p_where.add_argument(
        "--platforms", help="comma-separated platform names (default: all)"
    )
    p_where.add_argument(
        "--config", default="default", help="CI config (default: default)"
    )
    p_where.add_argument("--json", action="store_true")
    p_where.set_defaults(func=_cmd_where)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
