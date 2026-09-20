"""Dev-only archive checks and source correspondence, not build attestation."""
import hashlib
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(root, package, source, evidence):
    archive = package / "libeshkol_transformer_m3.a"
    members = subprocess.check_output(["ar", "t", str(archive)], text=True)
    if members != (root / "native/m3_package_archive_members.txt").read_text():
        raise ValueError("actual archive member inventory differs")
    (evidence / "archive-members.txt").write_text(members)
    member = evidence / "package-member.o"
    with member.open("wb") as output:
        subprocess.run(["ar", "p", str(archive), "m3_package.o"], stdout=output, check=True)
    if digest(member) != digest(package / "m3_package.o"):
        raise ValueError("archive member differs from adjacent package object")
    symbols = subprocess.check_output(
        ["nm", "--extern-only", "--defined-only", "--format=posix", str(member)], text=True
    )
    names = "".join(name + "\n" for name in sorted(line.split()[0] for line in symbols.splitlines()))
    (evidence / "actual-global-defined.txt").write_text(names)
    if names != (root / "native/m3_package_defined_symbols.txt").read_text():
        raise ValueError("actual archive global symbols differ")
    paths = set()
    for kind in ("source_closure", "native_source_closure"):
        paths.update(path for path in (root / f"native/m3_package_{kind}.txt").read_text().splitlines()
                     if not path.startswith(".deps/"))
    paths.update("lib/" + path for path in (root / "native/m3_package_facades.txt").read_text().splitlines())
    rows = ["path\tcurrent_sha256\tpackage_source_sha256\n"]
    for path in sorted(paths):
        current, prior = digest(root / path), digest(source / path)
        if current != prior:
            raise ValueError(f"package source differs: {path}")
        rows.append(f"{path}\t{current}\t{prior}\n")
    (evidence / "source-equality.tsv").write_text("".join(rows))
    (evidence / "source-provenance.txt").write_text(
        f"current_root={root}\npackage_source_root={source}\npackage={package}\n"
        f"archive_sha256={digest(archive)}\nmember_sha256={digest(member)}\n"
        "This is source correspondence and archive inventory evidence, not a fresh package build or cryptographic build attestation.\n"
    )
    print(f"G3 package actual archive/symbols PASS; {len(paths)} source files match")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise ValueError("usage: check_package.py ROOT PACKAGE PACKAGE_SOURCE_ROOT EVIDENCE")
        main(*(Path(arg).resolve(strict=True) for arg in sys.argv[1:]))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"G3 package verification failed: {error}")
