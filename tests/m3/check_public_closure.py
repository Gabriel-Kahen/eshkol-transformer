"""Reject public AOT dependencies outside one caller and the installed facades."""
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate(installed, source, depfile):
    installed, source, depfile = map(Path, (installed, source, depfile))
    for path in (installed, source):
        if not path.is_absolute() or str(path) != str(path.resolve(strict=True)):
            raise ValueError("public closure paths must be canonical and absolute")
    if not depfile.is_file() or not depfile.stat().st_size:
        raise ValueError("public compile-only depfile is missing or empty")
    _, separator, prerequisites = depfile.read_text().partition(":")
    if not separator:
        raise ValueError("public depfile lacks its target separator")
    dependencies = shlex.split(prerequisites.replace("\\\n", " "))
    facades = (ROOT / "native/m3_package_facades.txt").read_text().splitlines()
    allowed = {str(source)} | {str(installed / "facades" / name) for name in facades}
    if str(source) not in dependencies or len(dependencies) != len(set(dependencies)):
        raise ValueError("public depfile must name its caller once and omit duplicates")
    for dependency in dependencies:
        if dependency not in allowed:
            raise ValueError(f"unapproved public dependency: {dependency}")
        if str(Path(dependency).resolve(strict=True)) != dependency:
            raise ValueError(f"aliased public dependency: {dependency}")
    if set(dependencies) != allowed:
        raise ValueError("public depfile must contain the exact caller and all seven facades")
    return dependencies


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("usage: check_public_closure.py INSTALLED SOURCE DEPFILE")
        dependencies = validate(*sys.argv[1:])
    except (OSError, ValueError) as error:
        raise SystemExit(f"M3 public source closure rejected: {error}")
    print(f"M3 installed public source closure PASS: {len(dependencies)} files")
