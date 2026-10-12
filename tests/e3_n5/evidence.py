"""Deterministic N5 fixture digest and complete failure-preserving evidence seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entries(root: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []

    def walk(directory: Path) -> None:
        for item in sorted(os.scandir(directory), key=lambda entry: entry.name):
            path = Path(item.path)
            relative = path.relative_to(root).as_posix()
            mode = item.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                result.append({"path": relative, "type": "directory"})
                walk(path)
            elif stat.S_ISREG(mode):
                result.append({"path": relative, "type": "file",
                               "bytes": item.stat(follow_symlinks=False).st_size,
                               "sha256": sha256(path)})
            elif stat.S_ISLNK(mode):
                result.append({"path": relative, "type": "symlink",
                               "target": os.readlink(path)})
            else:
                result.append({"path": relative, "type": "nonregular",
                               "mode": stat.S_IFMT(mode)})

    walk(root)
    return result


def fixture_digest(root: Path) -> str:
    inventory = entries(root)
    if any(row["type"] not in ("file", "directory") for row in inventory):
        raise ValueError("fixture contains a symlink or nonregular entry")
    listing = "".join(
        f'{row["sha256"]}  ./{row["path"]}\n'
        for row in sorted(inventory, key=lambda item: str(item["path"]))
        if row["type"] == "file"
    )
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".n5-seal-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def seal(root: Path, command_status: int, stage: str) -> int:
    # Existing seal files are excluded so resealing a failed run is deterministic.
    excluded = {"SHA256SUMS", "INVENTORY.jsonl", "exit.status", "stage.txt"}
    inventory = []
    policy_ok = True
    for row in entries(root):
        if row["path"] in excluded:
            if row["type"] != "file":
                policy_ok = False
                inventory.append({**row, "path": "preseal/" + str(row["path"])})
        else:
            inventory.append(row)
            if row["type"] not in ("file", "directory"):
                policy_ok = False
    final_status = command_status if command_status else (0 if policy_ok else 1)
    _atomic_text(root / "exit.status", f"{final_status}\n")
    recorded_stage = stage if policy_ok else stage + "+inventory"
    _atomic_text(root / "stage.txt", recorded_stage + "\n")
    inventory.extend([
        {"path": "exit.status", "type": "file", "bytes": (root / "exit.status").stat().st_size,
         "sha256": sha256(root / "exit.status")},
        {"path": "stage.txt", "type": "file", "bytes": (root / "stage.txt").stat().st_size,
         "sha256": sha256(root / "stage.txt")},
    ])
    inventory.sort(key=lambda row: str(row["path"]))
    _atomic_text(root / "INVENTORY.jsonl",
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in inventory)
    )
    checksum_rows = []
    for row in inventory:
        if row["type"] == "file":
            relative = str(row["path"])
            if any(character in relative for character in "\\\n\r"):
                # The JSON inventory still seals this path and its byte digest.
                continue
            checksum_rows.append(f'{row["sha256"]}  {relative}\n')
    checksum_rows.append(f'{sha256(root / "INVENTORY.jsonl")}  INVENTORY.jsonl\n')
    _atomic_text(root / "SHA256SUMS", "".join(checksum_rows))
    return final_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture-digest")
    fixture.add_argument("root", type=Path)
    sealed = sub.add_parser("seal")
    sealed.add_argument("root", type=Path)
    sealed.add_argument("status", type=int)
    sealed.add_argument("stage")
    args = parser.parse_args()
    if args.command == "fixture-digest":
        print(fixture_digest(args.root))
    else:
        raise SystemExit(seal(args.root, args.status, args.stage))


if __name__ == "__main__":
    main()
