"""Deterministic N5 fixture digest and complete failure-preserving evidence seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


SEAL_FILES = ("exit.status", "stage.txt", "INVENTORY.jsonl", "SHA256SUMS")
COLLISION_FILE = "SEAL-COLLISION.json"
COLLISION_STATUS = 73


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


def _exclusive_text(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)


def _collision(root: Path, command_status: int, stage: str) -> int:
    collision = root / COLLISION_FILE
    if not os.path.lexists(collision):
        names = [name for name in SEAL_FILES if os.path.lexists(root / name)]
        record = {"command_status": command_status, "reserved_collisions": names,
                  "stage": stage}
        try:
            _exclusive_text(collision, json.dumps(record, sort_keys=True) + "\n")
        except FileExistsError:
            pass
    return COLLISION_STATUS


def seal(root: Path, command_status: int, stage: str) -> int:
    # A failed run's seal is evidence. Never overwrite it or invent a preseal path.
    if os.path.lexists(root / COLLISION_FILE) or any(
            os.path.lexists(root / name) for name in SEAL_FILES):
        return _collision(root, command_status, stage)
    inventory = entries(root)
    policy_ok = all(row["type"] in ("file", "directory") for row in inventory)
    final_status = command_status if command_status else (0 if policy_ok else 1)
    recorded_stage = stage if policy_ok else stage + "+inventory"
    try:
        _exclusive_text(root / "exit.status", f"{final_status}\n")
        _exclusive_text(root / "stage.txt", recorded_stage + "\n")
        inventory.extend([
            {"path": name, "type": "file", "bytes": (root / name).stat().st_size,
             "sha256": sha256(root / name)}
            for name in ("exit.status", "stage.txt")
        ])
        inventory.sort(key=lambda row: str(row["path"]))
        _exclusive_text(root / "INVENTORY.jsonl",
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
        _exclusive_text(root / "SHA256SUMS", "".join(checksum_rows))
    except FileExistsError:
        return _collision(root, command_status, stage)
    return final_status


def verify_inventory(root: Path) -> bool:
    try:
        for name in ("INVENTORY.jsonl", "SHA256SUMS"):
            if not stat.S_ISREG(os.lstat(root / name).st_mode):
                return False
        expected = [json.loads(line) for line in
                    (root / "INVENTORY.jsonl").read_text(encoding="utf-8").splitlines()]
        actual = [row for row in entries(root)
                  if row["path"] not in ("INVENTORY.jsonl", "SHA256SUMS")]
        actual.sort(key=lambda row: str(row["path"]))
        return expected == actual
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture-digest")
    fixture.add_argument("root", type=Path)
    verify = sub.add_parser("verify-inventory")
    verify.add_argument("root", type=Path)
    sealed = sub.add_parser("seal")
    sealed.add_argument("root", type=Path)
    sealed.add_argument("status", type=int)
    sealed.add_argument("stage")
    args = parser.parse_args()
    if args.command == "fixture-digest":
        print(fixture_digest(args.root))
    elif args.command == "verify-inventory":
        raise SystemExit(0 if verify_inventory(args.root) else 1)
    else:
        raise SystemExit(seal(args.root, args.status, args.stage))


if __name__ == "__main__":
    main()
