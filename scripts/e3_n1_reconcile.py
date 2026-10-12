"""Read-only, synthetic E3 N1 crash reconciliation; no production admission path."""

import json
import os
import stat
from dataclasses import dataclass

import e3_n1_broker as broker


@dataclass(frozen=True)
class Reconciliation:
    state: str  # HOLD or MATCHED; MATCHED is synthetic, not a deployment seal.
    reason: str
    attempt: str


def _directory(parent, name):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("directory-kind")
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                 os.O_CLOEXEC, dir_fd=parent)
    if not os.path.samestat(before, os.fstat(fd)):
        os.close(fd)
        raise ValueError("directory-inode-drift")
    return fd


def _base(config, attempt):
    return {"attempt": attempt, "pin": config.pin, "target": config.target,
            "stage": ".stage-" + attempt}


def _files_match(config, fd, files, complete):
    expected = {source.relative: source for source in config.sources}
    if not isinstance(files, list) or len(files) != len({item.get("path") for item in files
                                                           if isinstance(item, dict)}):
        return False
    if complete and len(files) != len(expected):
        return False
    paths = [item.get("path") for item in files if isinstance(item, dict)]
    if len(paths) != len(files) or paths != [source.relative for source in
                                              config.sources[:len(files)]]:
        return False
    reader = broker.Broker(config)
    present = set()
    for path in expected:
        try:
            reader._stage_file(fd, path)
            present.add(path)
        except FileNotFoundError:
            pass
    if present != {item["path"] for item in files}:
        return False
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            return False
        source = expected.get(item["path"])
        if source is None or (complete and item != {"path": source.relative,
                                                   "size": source.size,
                                                   "sha256": source.sha256}):
            return False
        if reader._stage_file(fd, item["path"]) != item:
            return False
    return True


def _duplicate_verified(parent, attempt, target):
    matches = []
    with os.scandir(parent) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            fd = _directory(parent, entry.name)
            try:
                try:
                    marker = broker._read_record(fd, "VERIFIED.json")
                except FileNotFoundError:
                    continue
                if marker.get("attempt") == attempt:
                    matches.append(entry.name)
            finally:
                os.close(fd)
    return matches != [target]


def _exact_target_tree(target, config):
    files = {"HOLD.json", "VERIFIED.json"} | {source.relative for source in config.sources}
    directories = {"/".join(path.split("/")[:index]) for path in files
                   for index in range(1, len(path.split("/")))}
    pending = [("", target)]
    opened = []
    try:
        seen_files = set()
        seen_directories = set()
        while pending:
            prefix, directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = prefix + entry.name
                    if path in directories and entry.is_dir(follow_symlinks=False):
                        child = _directory(directory, entry.name)
                        opened.append(child)
                        pending.append((path + "/", child))
                        seen_directories.add(path)
                    elif path in files and entry.is_file(follow_symlinks=False):
                        seen_files.add(path)
                    else:
                        return False
        return seen_files == files and seen_directories == directories
    finally:
        for fd in opened:
            os.close(fd)


def reconcile(config, attempt, receipt=None, seal=None):
    """Compare protected local evidence; MATCHED needs an external receipt and seal.

    This does not issue receipts, repair stages, establish durable receipt custody,
    or establish read-only filesystem custody. Every unreadable/ambiguous state
    remains HOLD. Inputs must come from a separately reviewed synthetic fixture.
    """
    hold = lambda reason: Reconciliation("HOLD", reason, attempt)
    try:
        config.validate()
        if len(attempt) != 32 or any(c not in "0123456789abcdef" for c in attempt):
            return hold("invalid-attempt")
        parent = os.open(config.parent, os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            base = _base(config, attempt)
            intent = broker._read_record(parent, ".attempt-" + attempt + ".json")
            if intent != dict(base, state="HOLD", reason="in-progress"):
                return hold("intent-mismatch")
            try:
                stage = _directory(parent, base["stage"])
            except FileNotFoundError:
                stage = None
            if stage is not None:
                try:
                    marker = broker._read_record(stage, "HOLD.json")
                    if (any(marker.get(key) != value for key, value in base.items()) or
                            marker.get("state") != "HOLD" or
                            not _files_match(config, stage, marker.get("files", []), False)):
                        return hold("stage-evidence-mismatch")
                    return hold("stage-present")
                finally:
                    os.close(stage)
            try:
                target = _directory(parent, config.target)
            except FileNotFoundError:
                return hold("target-absent")
            try:
                verified = broker._read_record(target, "VERIFIED.json")
                initial = broker._read_record(target, "HOLD.json")
                if initial != dict(base, state="HOLD", reason="in-progress"):
                    return hold("initial-marker-mismatch")
                info = os.fstat(target)
                expected = dict(base, state="VERIFIED", reason="checked",
                                device=info.st_dev, inode=info.st_ino)
                if any(verified.get(key) != value for key, value in expected.items()):
                    return hold("target-record-mismatch")
                if not _files_match(config, target, verified.get("files"), True):
                    return hold("target-files-mismatch")
                if not _exact_target_tree(target, config):
                    return hold("target-layout-mismatch")
            finally:
                os.close(target)
            if _duplicate_verified(parent, attempt, config.target):
                return hold("duplicate-target")
            if receipt is None:
                return hold("receipt-absent")
            if seal is None:
                return hold("seal-absent")
            body = json.loads(receipt)["body"]
            if body.get("attempt") != attempt or not broker.verify_receipt(
                    config, receipt, seal):
                return hold("receipt-or-seal-mismatch")
            return Reconciliation("MATCHED", "synthetic-evidence-matched", attempt)
        finally:
            os.close(parent)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return hold("evidence-unreadable")
