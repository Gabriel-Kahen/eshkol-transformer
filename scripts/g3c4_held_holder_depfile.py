"""Fail-closed Eshkol depfile parsing for the held-holder overlay gate."""

from pathlib import Path
import shlex


def dependencies(depfile: Path, root: Path, overrides: dict[Path, str]) -> list[str]:
    text = depfile.read_text().replace('\\\n', '')
    items = shlex.split(text.split(':', 1)[1])
    names = []
    observed = set()
    for item in items:
        path = Path(item)
        path = (path if path.is_absolute() else root / path).resolve(strict=True)
        if path in overrides:
            names.append(overrides[path])
            observed.add(path)
            continue
        assert path.is_relative_to(root), item
        name = path.relative_to(root).as_posix()
        assert name.endswith('.esk'), name
        assert name not in overrides.values(), name
        names.append(name)
    assert len(names) == len(set(names)), names
    assert observed == set(overrides), observed
    return names
