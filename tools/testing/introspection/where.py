"""Behavior 2: given a test name, which platforms does it run on?

For each platform we run the file's gutted suite (collector.status) and report, for
the tests matching the query, RUNS / SKIPPED(reason) / NOT_GENERATED. Device-generic
tests naturally appear under per-device names (TestFooCUDA::test_bar_cuda), so a query
like "test_bar" matches the right variant on each platform; a platform where no
variant exists at all is NOT_GENERATED (distinct from SKIPPED).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.testing.introspection import collector, platforms


if TYPE_CHECKING:
    from tools.testing.introspection.platforms import Job


def _match(query: str, test_id: str) -> bool:
    """test_id is 'Class::method'. Match a full id, a bare method, or a substring."""
    _, method = test_id.split("::", 1)
    return query in (test_id, method) or query in test_id


def _lookup_files(query: str, job: Job) -> list[str]:
    """Scan the selected universe (cache-backed) for files with a matching test."""
    selected = collector.select_files(job)["files"]
    raw = collector.collect(job, "enumerate", selected)
    files = []
    for f, r in raw.items():
        if "error" in r:
            continue
        if any(
            _match(query, f"{c}::{m}") for c, ms in r["classes"].items() for m in ms
        ):
            files.append(f)
    return sorted(files)


def where(
    query: str,
    file: str | None = None,
    platform_names: list[str] | None = None,
    config: str = "default",
) -> dict:
    plats = platform_names or list(platforms.REGISTRY)
    jobs = [platforms.get_job(f"{p}/{config}") for p in plats]
    files = (
        [file]
        if file
        else _lookup_files(query, platforms.get_job(f"linux-cpu/{config}"))
    )

    out: dict[str, dict] = {}
    for f in files:
        per_plat: dict[str, dict] = {}
        for j in jobs:
            try:
                st = collector.status(f, j)
            except Exception as e:
                per_plat[j.platform.name] = {"error": str(e).splitlines()[0]}
                continue
            runs = sorted(t for t in st["ran"] if _match(query, t))
            skipped = sorted((t, r) for t, r in st["skipped"] if _match(query, t))
            per_plat[j.platform.name] = {"runs": runs, "skipped": skipped}
        out[f] = per_plat
    return {"query": query, "platforms": plats, "files": out}
