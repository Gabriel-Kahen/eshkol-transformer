"""Reject installed E3 callers that import trusted or uninstalled source."""
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate(installed: Path, source: Path, depfile: Path) -> list[str]:
    installed, source, depfile = map(Path, (installed, source, depfile))
    for path in (installed, source):
        if not path.is_absolute() or str(path) != str(path.resolve(strict=True)):
            raise ValueError("public closure paths must be canonical and absolute")
    _, separator, prerequisites = depfile.read_text().partition(":")
    if not separator:
        raise ValueError("public depfile lacks its target separator")
    dependencies = shlex.split(prerequisites.replace("\\\n", " "))
    facades = (ROOT / "native/e3_diagnostic_public_package_facades.txt").read_text().splitlines()
    allowed = {str(source)} | {str(installed / "facades" / path) for path in facades}
    if dependencies.count(str(source)) != 1 or len(dependencies) != len(set(dependencies)):
        raise ValueError("public depfile must name its caller once and omit duplicates")
    for dependency in dependencies:
        if dependency not in allowed:
            raise ValueError(f"unapproved public dependency: {dependency}")
        if str(Path(dependency).resolve(strict=True)) != dependency:
            raise ValueError(f"aliased public dependency: {dependency}")
    if set(dependencies) != allowed:
        raise ValueError("public depfile does not contain the exact facade closure")
    return dependencies


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("usage: check_public_closure.py INSTALLED SOURCE DEPFILE")
        result = validate(*(Path(value) for value in sys.argv[1:]))
    except (OSError, ValueError) as error:
        raise SystemExit(f"E3 installed source closure rejected: {error}")
    print(f"E3 installed source closure PASS: {len(result)} files")
