"""Synthetic, source-only durable receipt protocol; no deployment admission."""

import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

import e3_n1_broker as broker
import e3_n1_reconcile as reconciler


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _parent_chain(path):
    if not path.is_absolute():
        raise ValueError("parent-not-absolute")
    parts = [Path(path.anchor)]
    for part in path.parts[1:]:
        parts.append(parts[-1] / part)
    chain = []
    for item in parts:
        info = os.lstat(item)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("parent-chain-kind")
        chain.append([info.st_dev, info.st_ino, info.st_mode])
    return chain


def release_descriptor(config, *, receipt_store: Path, image_id, compiler_head, compiler_tree,
                       source_head, source_tree):
    """Freeze exact release inputs; synthetic callers must identify fake pins."""
    config.validate()
    identities = (image_id, compiler_head, compiler_tree, source_head, source_tree)
    if not all(isinstance(value, str) and
               re.fullmatch(r"synthetic:[a-z0-9-]{1,48}", value)
               for value in identities):
        raise ValueError("non-synthetic-release-identity")
    return {"schema": "eshkol-e3-n1-synthetic-release-v2",
            "image_id": image_id, "compiler_head": compiler_head,
            "compiler_tree": compiler_tree, "source_head": source_head,
            "source_tree": source_tree, "target": config.target,
            "parent": str(config.parent), "parent_chain": _parent_chain(config.parent),
            "receipt_store": str(receipt_store),
            "receipt_store_chain": _parent_chain(receipt_store),
            "byte_cap": config.byte_cap,
            "seconds": config.seconds, "rss_cap": config.rss_cap,
            "seal_key_id": config.seal_key_id,
            "sources": [{"path": str(s.path), "relative": s.relative,
                         "sha256": s.sha256, "size": s.size,
                         "resolved": str(s.resolved),
                         "path_inodes": [list(row) for row in s.path_inodes]}
                        for s in config.sources]}


def release_pin(release):
    return _hash(broker._encoded(release))


def _validate_release(config, release, store):
    if not isinstance(release, dict) or set(release) != {
            "schema", "image_id", "compiler_head", "compiler_tree",
            "source_head", "source_tree", "target", "parent", "byte_cap",
            "seconds", "rss_cap", "seal_key_id", "sources", "parent_chain",
            "receipt_store", "receipt_store_chain"}:
        raise ValueError("release-fields")
    expected = release_descriptor(config, receipt_store=store,
        **{name: release[name] for name in (
        "image_id", "compiler_head", "compiler_tree", "source_head", "source_tree")})
    if release != expected or release_pin(release) != config.pin:
        raise ValueError("release-pin-mismatch")


def _open_pinned_store(release, store):
    fd = os.open(store, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if ([info.st_dev, info.st_ino, info.st_mode] !=
                release["receipt_store_chain"][-1] or
                _parent_chain(store) != release["receipt_store_chain"]):
            raise ValueError("receipt-store-inode-drift")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_exact(fd, name, limit=65536):
    before = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("record-kind-or-size")
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
    try:
        if not os.path.samestat(before, os.fstat(handle)):
            raise ValueError("record-inode-drift")
        data = os.read(handle, limit + 1)
        if len(data) != before.st_size or not os.path.samestat(
                before, os.stat(name, dir_fd=fd, follow_symlinks=False)):
            raise ValueError("record-path-drift")
        return data, before
    finally:
        os.close(handle)


def _write_noreplace(fd, name, data, checkpoint):
    temp = ".pending-" + secrets.token_hex(16)
    handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
    try:
        checkpoint("write:" + name)
        view = memoryview(data)
        while view:
            view = view[os.write(handle, view):]
        checkpoint("sync-file:" + name)
        os.fsync(handle)
    finally:
        os.close(handle)
    checkpoint("publish:" + name)
    broker._rename_noreplace(fd, temp, name)
    checkpoint("sync-dir:" + name)
    os.fsync(fd)
    return _read_exact(fd, name)


@dataclass(frozen=True)
class DurableResult:
    state: str
    reason: str
    attempt: str
    token: bytes | None = None


def issue(config, release, outcome, store: Path, checkpoint=lambda name: None):
    """Publish an attempt-keyed receipt outside export, then its commit record."""
    attempt = outcome.attempt
    try:
        _validate_release(config, release, store)
        if (outcome.state != "VERIFIED" or not outcome.receipt or
                len(attempt) != 32 or any(c not in "0123456789abcdef" for c in attempt)):
            raise ValueError("no-verified-outcome")
        receipt = json.loads(outcome.receipt)
        body = receipt["body"]
        if (broker._encoded(receipt) != outcome.receipt or
                body["attempt"] != attempt or body["pin"] != config.pin or
                not broker.hmac.compare_digest(receipt["hmac_sha256"],
                    broker.hmac.digest(config.signing_key, broker._encoded(body),
                                       "sha256").hex())):
            raise ValueError("receipt-authentication")
        # A readable VERIFIED marker alone never authorizes issuance.
        parent = os.open(config.parent, os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            intent = broker._read_record(parent, ".attempt-" + attempt + ".json")
            expected_intent = dict(reconciler._base(config, attempt),
                                   state="HOLD", reason="in-progress")
            if intent != expected_intent or reconciler._duplicate_verified(
                    parent, attempt, config.target):
                raise ValueError("intent-or-duplicate-target")
            target = reconciler._directory(parent, config.target)
            try:
                marker = broker._read_record(target, "VERIFIED.json")
                expected_marker = dict(expected_intent, state="VERIFIED", reason="checked",
                                       device=body["device"], inode=body["inode"],
                                       files=body["files"])
                if (marker != expected_marker or
                        broker._read_record(target, "HOLD.json") != expected_intent or
                        (os.fstat(target).st_dev, os.fstat(target).st_ino) !=
                        (body["device"], body["inode"]) or
                        not reconciler._files_match(config, target, body["files"], True) or
                        not reconciler._exact_target_tree(target, config)):
                    raise ValueError("target-evidence-mismatch")
            finally:
                os.close(target)
        finally:
            os.close(parent)
        fd = _open_pinned_store(release, store)
        try:
            name = "receipt-" + attempt + ".json"
            commit = "commit-" + attempt + ".json"
            if os.path.lexists(store / name) or os.path.lexists(store / commit):
                raise FileExistsError("attempt-already-published")
            payload = broker._encoded({"schema": "eshkol-e3-n1-durable-receipt-v2",
                "release": release, "release_sha256": release_pin(release),
                "receipt": receipt, "receipt_sha256": _hash(outcome.receipt)})
            actual, info = _write_noreplace(fd, name, payload, checkpoint)
            if actual != payload:
                raise ValueError("receipt-readback")
            token = secrets.token_bytes(32)
            marker = broker._encoded({"schema": "eshkol-e3-n1-receipt-commit-v2",
                "attempt": attempt, "receipt_sha256": _hash(payload),
                "device": info.st_dev, "inode": info.st_ino,
                "issuance_token_sha256": _hash(token)})
            if _write_noreplace(fd, commit, marker, checkpoint)[0] != marker:
                raise ValueError("commit-readback")
            checkpoint("before-issue-return")
            return DurableResult("DURABLE", "checked", attempt, token)
        finally:
            os.close(fd)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError) as error:
        return DurableResult("HOLD", type(error).__name__ + ":" + str(error), attempt)


@dataclass(frozen=True)
class SealRef:
    path: Path
    sha256: str
    device: int
    inode: int


def reconcile_durable(config, release, attempt, store: Path, receipt: bytes | None,
                      token: bytes | None, seal: SealRef | None):
    """Cross-check externally held receipt and seal against durable evidence."""
    hold = lambda reason: reconciler.Reconciliation("HOLD", reason, attempt)
    try:
        _validate_release(config, release, store)
        if len(attempt) != 32 or any(c not in "0123456789abcdef" for c in attempt):
            return hold("invalid-attempt")
        if receipt is None:
            return hold("external-receipt-absent")
        if not isinstance(token, bytes) or len(token) != 32:
            return hold("issuance-token-absent")
        fd = _open_pinned_store(release, store)
        try:
            raw, info = _read_exact(fd, "receipt-" + attempt + ".json")
            committed, _ = _read_exact(fd, "commit-" + attempt + ".json")
        finally:
            os.close(fd)
        record = json.loads(raw)
        commit = json.loads(committed)
        if (broker._encoded(record) != raw or broker._encoded(commit) != committed or
                set(record) != {"schema", "release", "release_sha256", "receipt",
                                "receipt_sha256"} or
                record["schema"] != "eshkol-e3-n1-durable-receipt-v2" or
                record["release"] != release or
                record["release_sha256"] != config.pin or
                _hash(broker._encoded(record["receipt"])) != record["receipt_sha256"] or
                commit != {"schema": "eshkol-e3-n1-receipt-commit-v2",
                           "attempt": attempt, "receipt_sha256": _hash(raw),
                           "device": info.st_dev, "inode": info.st_ino,
                           "issuance_token_sha256": _hash(token)}):
            return hold("durable-receipt-mismatch")
        if receipt != broker._encoded(record["receipt"]):
            return hold("external-receipt-mismatch")
        if seal is None:
            return hold("seal-absent")
        if not seal.path.is_absolute() or seal.path.is_symlink():
            return hold("seal-path")
        parent = os.open(seal.path.parent, os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            seal_bytes, identity = _read_exact(parent, seal.path.name)
        finally:
            os.close(parent)
        if ((identity.st_dev, identity.st_ino) != (seal.device, seal.inode) or
                _hash(seal_bytes) != seal.sha256):
            return hold("seal-identity-mismatch")
        return reconciler.reconcile(config, attempt, receipt, seal_bytes)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return hold("durable-evidence-unreadable")
