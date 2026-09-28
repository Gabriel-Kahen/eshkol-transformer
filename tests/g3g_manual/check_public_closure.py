"""Only the installed G3-G facades may enter a public AOT caller."""
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate(installed: Path, source: Path, depfile: Path) -> list[str]:
    for path in (installed, source):
        if not path.is_absolute() or str(path) != str(path.resolve(strict=True)):
            raise ValueError("caller and installation must be canonical absolute paths")
    if not depfile.is_file() or not depfile.stat().st_size:
        raise ValueError("public caller depfile is missing")
    _, separator, prerequisites = depfile.read_text().partition(":")
    if not separator:
        raise ValueError("public caller depfile lacks a target")
    dependencies = shlex.split(prerequisites.replace("\\\n", " "))
    facades = (ROOT / "native/g3g_manual_package_facades.txt").read_text().splitlines()
    allowed = {str(source)} | {str(installed / "facades" / name) for name in facades}
    if str(source) not in dependencies or len(dependencies) != len(set(dependencies)):
        raise ValueError("caller is absent or dependencies repeat")
    if str(installed / "facades/transformer/generation.esk") not in dependencies:
        raise ValueError("G3-G facade is absent")
    for dependency in dependencies:
        if dependency not in allowed:
            raise ValueError(f"unapproved public dependency: {dependency}")
        if str(Path(dependency).resolve(strict=True)) != dependency:
            raise ValueError(f"aliased public dependency: {dependency}")
    return dependencies


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("usage: check_public_closure.py INSTALLED SOURCE DEPFILE")
        result = validate(*(Path(value) for value in sys.argv[1:]))
    except (OSError, ValueError) as error:
        raise SystemExit(f"G3-G public closure rejected: {error}")
    print(f"G3-G public closure PASS: {len(result)} source files")
