"""Synthetic E3 N1 custody broker candidate. No production pin or CLI entrypoint.

Only the broker process may construct Config. Its parent and sources must come
from a separately reviewed release, never from an acquisition caller.
"""

import ctypes
import hashlib
import hmac
import json
import os
import resource
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path


RENAME_NOREPLACE = 1
_renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
_renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint)
_renameat2.restype = ctypes.c_int


def _rename_noreplace(fd, old, new):
    if _renameat2(fd, old.encode(), fd, new.encode(), RENAME_NOREPLACE):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _encoded(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _record(body):
    return _encoded({"schema": 1, "sha256": _digest(_encoded(body)), "body": body})


def _read_record(fd, name):
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            raise ValueError("record-kind-or-size")
        data = os.read(handle, 65537)
    finally:
        os.close(handle)
    value = json.loads(data)
    if value != json.loads(_record(value["body"])):
        raise ValueError("record-checksum")
    return value["body"]


class SyncFailure(OSError):
    pass


@dataclass(frozen=True)
class Source:
    path: Path
    relative: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Config:
    parent: Path
    pin: str
    target: str
    sources: tuple[Source, ...]
    byte_cap: int
    seconds: float
    rss_cap: int
    signing_key: bytes

    def validate(self):
        if (len(self.pin) != 64 or any(c not in "0123456789abcdef" for c in self.pin)
                or not self.target or "/" in self.target or self.target in (".", "..")
                or not self.sources or not self.signing_key or
                min(self.byte_cap, self.seconds, self.rss_cap) <= 0):
            raise ValueError("invalid-frozen-config")
        names = set()
        for source in self.sources:
            name = source.relative
            valid = name in ("oci-layout", "index.json") or (
                name.startswith("blobs/sha256/") and len(name) == 77 and
                all(c in "0123456789abcdef" for c in name[13:]))
            if (not valid or name in names or source.size < 0 or
                    len(source.sha256) != 64 or
                    any(c not in "0123456789abcdef" for c in source.sha256) or
                    name.startswith("blobs/sha256/") and name[13:] != source.sha256):
                raise ValueError("invalid-source-descriptor")
            names.add(name)
        if sum(s.size for s in self.sources) > self.byte_cap:
            raise ValueError("byte-cap")


@dataclass(frozen=True)
class Outcome:
    state: str
    attempt: str
    reason: str
    receipt: bytes | None = None


class Broker:
    def __init__(self, config: Config, checkpoint=lambda name: None):
        config.validate()
        self.config = config
        self.checkpoint = checkpoint

    def _sync(self, fd, label):
        try:
            self.checkpoint(label)
            os.fsync(fd)
        except OSError as error:
            raise SyncFailure(str(error)) from error

    def _write(self, fd, name, content, exclusive=False):
        temp = ".tmp-" + secrets.token_hex(12)
        handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
        try:
            self.checkpoint("write:" + name)
            view = memoryview(content)
            while view:
                view = view[os.write(handle, view):]
            self._sync(handle, "sync-file:" + name)
        finally:
            os.close(handle)
        try:
            if exclusive:
                _rename_noreplace(fd, temp, name)
            else:
                os.replace(temp, name, src_dir_fd=fd, dst_dir_fd=fd)
            self._sync(fd, "sync-dir:" + name)
        except BaseException:
            # A failed cleanup is handled by the caller as indeterminate.
            try:
                os.unlink(temp, dir_fd=fd)
            except FileNotFoundError:
                pass
            raise

    def _copy(self, stage, deadline):
        hashes = []
        for source in self.config.sources:
            self.checkpoint("before-read:" + source.relative)
            before = os.stat(source.path, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_size != source.size:
                raise ValueError("source-kind-or-size")
            src = os.open(source.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                if _identity(before) != _identity(os.fstat(src)):
                    raise ValueError("source-open-drift")
                parts = source.relative.split("/")
                directory = stage
                opened = []
                try:
                    for part in parts[:-1]:
                        try:
                            os.mkdir(part, 0o700, dir_fd=directory)
                        except FileExistsError:
                            pass
                        child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                        os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                        opened.append(child)
                        directory = child
                    dst = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                  os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
                    try:
                        digest = hashlib.sha256()
                        count = 0
                        while block := os.read(src, 65536):
                            self.checkpoint("read-chunk:" + source.relative)
                            count += len(block)
                            if (count > source.size or time.monotonic() > deadline or
                                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >
                                    self.config.rss_cap):
                                raise ValueError("resource-cap")
                            digest.update(block)
                            view = memoryview(block)
                            while view:
                                view = view[os.write(dst, view):]
                        self._sync(dst, "sync-copy:" + source.relative)
                    finally:
                        os.close(dst)
                    for fd in reversed(opened):
                        self._sync(fd, "sync-copy-dir:" + source.relative)
                finally:
                    for fd in reversed(opened):
                        os.close(fd)
                after = os.stat(source.path, follow_symlinks=False)
                if (_identity(before) != _identity(after) or
                        _identity(before) != _identity(os.fstat(src)) or
                        after.st_size != source.size or count != source.size or
                        digest.hexdigest() != source.sha256):
                    raise ValueError("source-drift")
                hashes.append({"path": source.relative, "size": count,
                               "sha256": digest.hexdigest()})
            finally:
                os.close(src)
        self._sync(stage, "sync-stage")
        return hashes

    def attempt(self, pin, target):
        if pin != self.config.pin or target != self.config.target:
            return Outcome("HOLD", "", "unreviewed-request")
        attempt = secrets.token_hex(16)
        stage_name = ".stage-" + attempt
        intent_name = ".attempt-" + attempt + ".json"
        body = {"attempt": attempt, "pin": pin, "target": target,
                "stage": stage_name, "state": "HOLD", "reason": "in-progress"}
        parent = os.open(self.config.parent, os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
        stage = None
        created = renamed = marked = marker_attempted = initial_durable = False
        try:
            try:
                self._write(parent, intent_name, _record(body), exclusive=True)
            except Exception:
                return Outcome("HOLD", attempt, "intent-failure")
            try:
                self.checkpoint("mkdir-stage")
                os.mkdir(stage_name, 0o700, dir_fd=parent)
                created = True
                self.checkpoint("open-stage")
                stage = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY |
                                os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
                if not os.path.samestat(os.stat(stage_name, dir_fd=parent,
                                                follow_symlinks=False), os.fstat(stage)):
                    raise ValueError("stage-inode")
                marker_attempted = True
                self._write(stage, "HOLD.json", _record(body))
                marked = True
                self._sync(parent, "sync-initial-parent")
                initial_durable = True
                hashes = self._copy(stage, time.monotonic() + self.config.seconds)
                self.checkpoint("rename-target")
                _rename_noreplace(parent, stage_name, target)
                renamed = True
                if not os.path.samestat(os.stat(target, dir_fd=parent,
                                                follow_symlinks=False), os.fstat(stage)):
                    raise ValueError("target-inode")
                for item in hashes:
                    self._readback(stage, item)
                self._sync(parent, "sync-renamed-parent")
                verified = dict(body, state="VERIFIED", reason="checked",
                                inode=os.fstat(stage).st_ino, device=os.fstat(stage).st_dev,
                                files=hashes)
                self._write(stage, "VERIFIED.json", _record(verified))
                self._sync(stage, "sync-verified-stage")
                self._sync(parent, "sync-verified-parent")
                receipt_body = {"schema": 1, "attempt": attempt, "target": target,
                                "pin": pin, "device": verified["device"],
                                "inode": verified["inode"], "files": hashes}
                signature = hmac.digest(self.config.signing_key, _encoded(receipt_body), "sha256")
                receipt = _encoded({"body": receipt_body, "hmac_sha256": signature.hex()})
                return Outcome("VERIFIED", attempt, "checked", receipt)
            except Exception as error:
                reason = type(error).__name__ + ":" + str(error)
                indeterminate = (marker_attempted and not initial_durable) or isinstance(error, SyncFailure)
                if renamed:
                    try:
                        self.checkpoint("retract-target")
                        _rename_noreplace(parent, target, stage_name)
                        renamed = False
                        self._sync(parent, "sync-retracted-parent")
                    except Exception:
                        indeterminate = True
                if stage is not None and marked and not renamed:
                    try:
                        self._write(stage, "HOLD.json", _record(dict(body, reason=reason)))
                    except Exception:
                        indeterminate = True
                if created and stage is None:
                    try:
                        os.rmdir(stage_name, dir_fd=parent)
                        self._sync(parent, "sync-cleanup-parent")
                    except Exception:
                        indeterminate = True
                return Outcome("INDETERMINATE" if indeterminate else "HOLD", attempt, reason)
        finally:
            if stage is not None:
                os.close(stage)
            os.close(parent)

    def _readback(self, stage, item):
        directory = stage
        opened = []
        try:
            parts = item["path"].split("/")
            for part in parts[:-1]:
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                    os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                opened.append(directory)
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=directory)
            try:
                info = os.fstat(fd)
                digest = hashlib.sha256()
                while block := os.read(fd, 65536):
                    digest.update(block)
                if (not stat.S_ISREG(info.st_mode) or info.st_size != item["size"] or
                        digest.hexdigest() != item["sha256"]):
                    raise ValueError("target-readback")
            finally:
                os.close(fd)
        finally:
            for fd in reversed(opened):
                os.close(fd)


def verify_receipt(config, receipt, seal, seal_key):
    """Consumer admission; seal is separately signed over the exact file list."""
    try:
        value = json.loads(receipt)
        body = value["body"]
        if (not hmac.compare_digest(value["hmac_sha256"], hmac.digest(
                config.signing_key, _encoded(body), "sha256").hex()) or
                body["pin"] != config.pin or body["target"] != config.target):
            return False
        seal_value = json.loads(seal)
        if (not hmac.compare_digest(seal_value["hmac_sha256"], hmac.digest(
                seal_key, _encoded(seal_value["body"]), "sha256").hex()) or
                seal_value["body"] != body):
            return False
        parent = os.open(config.parent, os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            target = os.open(body["target"], os.O_RDONLY | os.O_DIRECTORY |
                             os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            try:
                info = os.fstat(target)
                if (info.st_dev, info.st_ino) != (body["device"], body["inode"]):
                    return False
                verified = _read_record(target, "VERIFIED.json")
                if verified["attempt"] != body["attempt"] or verified["files"] != body["files"]:
                    return False
                for file in body["files"]:
                    path = file["path"]
                    if path not in ("oci-layout", "index.json") and not (
                            path.startswith("blobs/sha256/") and len(path) == 77):
                        return False
                    directory = target
                    opened = []
                    try:
                        parts = path.split("/")
                        for part in parts[:-1]:
                            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                                os.O_NOFOLLOW | os.O_CLOEXEC,
                                                dir_fd=directory)
                            opened.append(directory)
                        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW |
                                     os.O_CLOEXEC, dir_fd=directory)
                        try:
                            info = os.fstat(fd)
                            if not stat.S_ISREG(info.st_mode) or info.st_size != file["size"]:
                                return False
                            digest = hashlib.sha256()
                            while block := os.read(fd, 65536):
                                digest.update(block)
                            if digest.hexdigest() != file["sha256"]:
                                return False
                        finally:
                            os.close(fd)
                    finally:
                        for fd in reversed(opened):
                            os.close(fd)
                return True
            finally:
                os.close(target)
        finally:
            os.close(parent)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return False
