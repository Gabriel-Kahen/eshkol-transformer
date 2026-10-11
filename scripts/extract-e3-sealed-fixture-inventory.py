#!/usr/bin/env python3
"""Extract observations; never synthesize admitted manifests from expectations."""

from pathlib import Path
import argparse
import re
import shutil
import sys


def source_closure(repo: Path, generated: Path, depfile: Path) -> bytes:
    raw = depfile.read_text()
    if not raw or not re.match(r"^[^:\n]+:", raw):
        raise ValueError("missing depfile target")
    words = raw.split(":", 1)[1].replace("\\\n", " ").split()
    substitutions = {
        generated / "e3_d2_dataset.esk": "@BUILD@/source/e3_d2_dataset.esk",
        generated / "i2_wave2_root.esk": "@BUILD@/source/i2_wave2_root.esk",
        generated / "m3t_transport_extension.esk":
            "@BUILD@/source/m3t_transport_extension.esk",
        generated / "transformer/module.esk": "@BUILD@/source/transformer/module.esk",
    }
    rows = []
    for word in words:
        path = Path(word)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError(f"nonabsolute or symlinked depfile input: {word}")
        if path in substitutions:
            rows.append(substitutions[path])
        else:
            try:
                rows.append(str(path.relative_to(repo)))
            except ValueError as exc:
                raise ValueError(f"foreign depfile input: {word}") from exc
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("empty or duplicate source closure")
    return ("\n".join(rows) + "\n").encode()


def object_inventory(evidence: Path, output: Path) -> None:
    names = ("private.d", "source-closure.txt", "native-source-closure.txt",
             "native-objects.txt", "global-defined.txt", "undefined.txt",
             "public-strings.txt", "package-exports.txt", "nm.txt",
             "readelf-symbols.txt", "link.map")
    if output.exists() or output.is_symlink():
        raise ValueError("inventory output must be fresh")
    if not evidence.is_dir() or evidence.is_symlink():
        raise ValueError("object evidence is absent or symlinked")
    missing = [name for name in names if not (evidence / name).is_file()
               or (evidence / name).is_symlink()]
    if missing:
        raise ValueError("incomplete emitted object evidence: " + ", ".join(missing))
    depfiles = evidence / "native-depfiles"
    if not depfiles.is_dir() or depfiles.is_symlink():
        raise ValueError("native depfiles absent")
    entries = sorted(depfiles.iterdir())
    if not entries or any(not item.is_file() or item.is_symlink() or
                          item.suffix != ".d" for item in entries):
        raise ValueError("unexpected native depfile entries")
    output.mkdir()
    for name in names:
        shutil.copyfile(evidence / name, output / name)
    (output / "native-depfiles").mkdir()
    for item in entries:
        shutil.copyfile(item, output / "native-depfiles" / item.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    source = sub.add_parser("source")
    for arg in ("repo", "generated", "depfile", "output"):
        source.add_argument(arg, type=Path)
    objects = sub.add_parser("objects")
    objects.add_argument("evidence", type=Path)
    objects.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.kind == "source":
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("source inventory output must be fresh")
            args.output.write_bytes(source_closure(args.repo, args.generated,
                                                  args.depfile))
        else:
            object_inventory(args.evidence, args.output)
    except (OSError, ValueError) as exc:
        print(f"inventory rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
