#!/usr/bin/env python3
"""Hash the explicit Eshkol source closure used by an operational entry."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


FORM = re.compile(r"\((?:load|import|require)\s+([^)]*)\)")


def candidates(name: str) -> tuple[str, ...]:
    is_path = (
        name.startswith(("/", "./", "../"))
        or "/" in name
        or name.endswith(".esk")
    )
    if not is_path:
        return (name.replace(".", "/") + ".esk",)
    return (name,) if name.endswith(".esk") else (name, name + ".esk")


def resolve(name: str, source: Path, roots: tuple[Path, ...]) -> Path | None:
    search = (source.parent, *roots)
    if name.startswith("/"):
        search = (Path("/"),)
    for directory in search:
        for candidate in candidates(name):
            path = (directory / candidate).resolve()
            if path.is_file():
                return path
    return None


def references(source: Path) -> tuple[str, ...]:
    result: list[str] = []
    for clause in FORM.findall(source.read_text(encoding="utf-8")):
        for token in clause.split():
            token = token.lstrip("('\"").rstrip(")\"")
            if token:
                result.append(token)
    return tuple(result)


def closure(entry: Path, roots: tuple[Path, ...]) -> set[Path]:
    seen: set[Path] = set()

    def visit(source: Path) -> None:
        source = source.resolve()
        if source in seen:
            return
        seen.add(source)
        for reference in references(source):
            dependency = resolve(reference, source, roots)
            if dependency is not None:
                visit(dependency)

    visit(entry)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--exclude-entry", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    roots = tuple(
        (project / path).resolve()
        for path in (
            "internal/p1/lib",
            "internal/c1/lib",
            "internal/t2/lib",
            "internal/t1/lib",
            "internal/d2/lib",
            "tests/c2",
            "src",
            "lib",
            "native",
        )
    )
    entry = args.entry.resolve()
    sources = closure(entry, roots)
    if args.exclude_entry:
        sources.remove(entry)
    for source in sorted(sources):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        try:
            display = source.relative_to(project)
        except ValueError:
            display = source
        print(f"{digest}  {display}")


if __name__ == "__main__":
    main()
