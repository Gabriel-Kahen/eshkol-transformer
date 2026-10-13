#!/usr/bin/env python3
"""Bound host streams, wall time, and guest output while running one Docker client."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import quote_from_bytes


SCAN_MESSAGES = {
    "traversal-bound": "guest output traversal bound exceeded",
    "directory-open-error": "guest output directory could not be opened without following links",
    "directory-replaced": "guest output directory identity changed during traversal",
    "watch-error": "guest output move watch could not be established or read",
    "tree-moved": "guest output tree moved during traversal",
    "scandir-error": "guest output directory scan failed",
    "stat-error": "guest output entry stat failed",
    "symlink-entry": "guest output contains a symlink",
    "nonregular-entry": "guest output contains an indirect or special entry",
    "hardlinked-entry": "guest output contains a hardlinked regular entry",
    "unclassified-error": "guest output traversal failed",
}

# Linux-only: there is no path-based fallback when descriptor/no-follow
# traversal is unavailable. O_PATH obtains metadata without opening file data
# or blocking on a FIFO, and O_NOFOLLOW keeps symlinks as entries to reject.
DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
ENTRY_OPEN_FLAGS = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC

# Structural moves of regular files or directories can invalidate a live
# measurement even when opened descriptors retain the expected inode.
MOVE_MASK = 0x00000040 | 0x00000080 | 0x00000400 | 0x00000800
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.inotify_init1.argtypes = (ctypes.c_int,)
LIBC.inotify_init1.restype = ctypes.c_int
LIBC.inotify_add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
LIBC.inotify_add_watch.restype = ctypes.c_int


class OutputScanFault(ValueError):
    """A fixed diagnostic with guest-relative provenance and no raw OS text."""

    def __init__(self, code: str, parts: tuple[str, ...],
                 error_class: str = "ValueError", errno_value: int | None = None):
        self.code = code
        self.parts = parts
        self.error_class = error_class
        self.errno_value = errno_value
        super().__init__(SCAN_MESSAGES[code])


def safe_guest_path(parts: tuple[str, ...]) -> str:
    if not parts:
        return "."
    encoded = []
    try:
        for part in parts:
            if not part or part in (".", "..") or "/" in part:
                return "<invalid-relative-path>"
            encoded.append(quote_from_bytes(os.fsencode(part), safe="-._~"))
    except (TypeError, ValueError):
        return "<unavailable-relative-path>"
    path = "/".join(encoded)
    return path if len(path) <= 256 else path[:253] + "..."


def safe_scan_detail(error: OSError | ValueError, phase: str) -> tuple[str, ...]:
    if isinstance(error, OutputScanFault):
        error_class, code, parts, number = (
            error.error_class, error.code, error.parts, error.errno_value)
    else:
        error_class = "OSError" if isinstance(error, OSError) else "ValueError"
        code, parts, number = "unclassified-error", (), None
    errno_text = (str(number) if isinstance(number, int)
                  and 0 <= number <= 9999 else "none")
    return (phase, error_class, code, safe_guest_path(parts),
            SCAN_MESSAGES[code], errno_text)


def open_directory_parts(anchor_fd: int, parts: tuple[str, ...]) -> int:
    current = os.dup(anchor_fd)
    try:
        for part in parts:
            if not part or part in (".", "..") or "/" in part:
                raise ValueError("invalid directory component")
            child = os.open(part, DIRECTORY_OPEN_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


class MoveWatchError(OSError):
    """A move watch could not be established without weakening the scan."""


class MoveWatch:
    """Keep Linux move watches on every directory seen in one bounded scan."""

    def __init__(self):
        self.fd = LIBC.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            number = ctypes.get_errno()
            raise MoveWatchError(number, os.strerror(number))
        self.watches = {}

    def add(self, directory_fd: int, parts: tuple[str, ...],
            child: str | None = None) -> None:
        # The proc-fd magic link identifies an already opened inode. It is
        # never a traversal path or a source of guest path text.
        path = f"/proc/self/fd/{directory_fd}".encode("ascii")
        watch = LIBC.inotify_add_watch(self.fd, path, MOVE_MASK)
        if watch < 0:
            number = ctypes.get_errno()
            raise MoveWatchError(number, os.strerror(number))
        # None means every structural move in this output directory matters.
        # Root-path ancestor watches care only about the next component.
        old = self.watches.get(watch)
        if old is None and watch in self.watches:
            return
        if child is None:
            self.watches[watch] = (parts, None)
        else:
            names = set() if old is None else set(old[1])
            names.add(os.fsencode(child))
            self.watches[watch] = (parts, names)

    def first_move(self) -> tuple[str, ...] | None:
        consumed = 0
        while consumed <= 65536:
            try:
                event = os.read(self.fd, 4096)
            except BlockingIOError:
                return None
            if not event:
                raise OSError("move watch unexpectedly closed")
            consumed += len(event)
            offset = 0
            while offset < len(event):
                if len(event) - offset < 16:
                    raise OSError("move watch event truncated")
                watch, mask, _, length = struct.unpack_from("iIII", event, offset)
                offset += 16
                if offset + length > len(event):
                    raise OSError("move watch name truncated")
                name = event[offset:offset + length].split(b"\0", 1)[0]
                offset += length
                if watch == -1 or watch not in self.watches:
                    return ()
                parts, names = self.watches[watch]
                if mask & (0x00000400 | 0x00000800 | 0x00004000 | 0x00008000):
                    return parts
                if names is None or name in names:
                    return parts
        raise OSError("move watch event bound exceeded")

    def close(self) -> None:
        os.close(self.fd)


def open_output_root(root: Path, watcher: MoveWatch | None = None) -> int:
    # Walk every root component without following an ancestor symlink. The
    # returned descriptor, not the path string, is the traversal authority.
    parts = root.parts
    anchor = "/" if root.is_absolute() else "."
    if root.is_absolute():
        parts = parts[1:]
    anchor_fd = os.open(anchor, DIRECTORY_OPEN_FLAGS)
    try:
        current = os.dup(anchor_fd)
        try:
            for part in parts:
                if not part or part in (".", "..") or "/" in part:
                    raise ValueError("invalid directory component")
                if watcher is not None:
                    watcher.add(current, (), child=part)
                child = os.open(part, DIRECTORY_OPEN_FLAGS, dir_fd=current)
                os.close(current)
                current = child
            return current
        except BaseException:
            os.close(current)
            raise
    finally:
        os.close(anchor_fd)


def output_bytes(root: Path, max_entries: int = 50000,
                 max_depth: int = 32, max_seconds: float = 2.0) -> int:
    total = 0
    count = 0
    deadline = time.monotonic() + max_seconds
    try:
        watcher = MoveWatch()
    except OSError as error:
        raise OutputScanFault("watch-error", (), "OSError", error.errno) from error
    try:
        root_fd = open_output_root(root, watcher)
    except MoveWatchError as error:
        watcher.close()
        raise OutputScanFault("watch-error", (), "OSError", error.errno) from error
    except (OSError, ValueError) as error:
        watcher.close()
        error_class = "OSError" if isinstance(error, OSError) else "ValueError"
        raise OutputScanFault("directory-open-error", (), error_class,
                              getattr(error, "errno", None)) from error
    try:
        root_info = os.fstat(root_fd)
        root_identity = (root_info.st_dev, root_info.st_ino)
        try:
            watcher.add(root_fd, ())
        except OSError as error:
            raise OutputScanFault("watch-error", (), "OSError", error.errno) from error
        pending = [((), 0, (root_info.st_dev, root_info.st_ino))]
        seen = []
        while pending:
            parts, depth, expected_identity = pending.pop()
            if depth > max_depth or time.monotonic() > deadline:
                raise OutputScanFault("traversal-bound", parts)
            try:
                directory_fd = open_directory_parts(root_fd, parts)
            except (OSError, ValueError) as error:
                error_class = "OSError" if isinstance(error, OSError) else "ValueError"
                raise OutputScanFault("directory-open-error", parts, error_class,
                                      getattr(error, "errno", None)) from error
            try:
                opened = os.fstat(directory_fd)
                if (opened.st_dev, opened.st_ino) != expected_identity:
                    raise OutputScanFault("directory-replaced", parts)
                if parts:
                    try:
                        watcher.add(directory_fd, parts)
                    except OSError as error:
                        raise OutputScanFault("watch-error", parts, "OSError",
                                              error.errno) from error
                seen.append((parts, expected_identity))
                try:
                    with os.scandir(directory_fd) as entries:
                        for entry in entries:
                            entry_parts = (*parts, entry.name)
                            count += 1
                            if count > max_entries or time.monotonic() > deadline:
                                raise OutputScanFault("traversal-bound", entry_parts)
                            try:
                                entry_fd = os.open(entry.name, ENTRY_OPEN_FLAGS,
                                                   dir_fd=directory_fd)
                                try:
                                    info = os.fstat(entry_fd)
                                finally:
                                    os.close(entry_fd)
                            except OSError as error:
                                raise OutputScanFault(
                                    "stat-error", entry_parts, "OSError", error.errno
                                ) from error
                            if stat.S_ISDIR(info.st_mode):
                                pending.append((entry_parts, depth + 1,
                                                (info.st_dev, info.st_ino)))
                            elif stat.S_ISREG(info.st_mode):
                                if info.st_nlink != 1:
                                    raise OutputScanFault("hardlinked-entry", entry_parts)
                                total += info.st_size
                            elif stat.S_ISLNK(info.st_mode):
                                raise OutputScanFault("symlink-entry", entry_parts)
                            else:
                                raise OutputScanFault("nonregular-entry", entry_parts)
                except OSError as error:
                    raise OutputScanFault(
                        "scandir-error", parts, "OSError", error.errno
                    ) from error
            finally:
                os.close(directory_fd)
        # Held directory descriptors can be reparented after opening. Verify
        # the named root and all scanned directories still identify the same
        # inodes, then reject even a transient move-out/move-back event.
        try:
            fresh_root = open_output_root(root)
        except (OSError, ValueError) as error:
            error_class = "OSError" if isinstance(error, OSError) else "ValueError"
            raise OutputScanFault("directory-open-error", (), error_class,
                                  getattr(error, "errno", None)) from error
        try:
            info = os.fstat(fresh_root)
            if (info.st_dev, info.st_ino) != root_identity:
                raise OutputScanFault("directory-replaced", ())
            for parts, expected_identity in seen:
                if time.monotonic() > deadline:
                    raise OutputScanFault("traversal-bound", parts)
                try:
                    directory_fd = open_directory_parts(fresh_root, parts)
                except (OSError, ValueError) as error:
                    error_class = "OSError" if isinstance(error, OSError) else "ValueError"
                    raise OutputScanFault("directory-open-error", parts, error_class,
                                          getattr(error, "errno", None)) from error
                try:
                    info = os.fstat(directory_fd)
                    if (info.st_dev, info.st_ino) != expected_identity:
                        raise OutputScanFault("directory-replaced", parts)
                finally:
                    os.close(directory_fd)
        finally:
            os.close(fresh_root)
        try:
            moved = watcher.first_move()
        except OSError as error:
            raise OutputScanFault("watch-error", (), "OSError", error.errno) from error
        if moved is not None:
            raise OutputScanFault("tree-moved", moved)
    finally:
        watcher.close()
        os.close(root_fd)
    return total


def group_absent(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def run(root: Path, output: Path, cap: int, output_cap: int,
        wall: int, command: list[str]) -> int:
    if (not root.is_absolute() or not root.is_dir() or root.is_symlink()
            or not output.is_dir() or output.is_symlink()
            or cap <= 0 or output_cap <= 0 or wall <= 0
            or not command or not Path(command[0]).is_absolute()):
        raise ValueError("invalid bounded-run authority")
    exceeded = threading.Event()
    counts = [0, 0]
    truncated = [False, False]
    first_output_error = None
    with (root / "docker.stdout").open("xb") as stdout, \
         (root / "docker.stderr").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)

        def pump(pipe, target, index: int) -> None:
            while data := pipe.read(65536):
                room = max(0, cap - counts[index])
                kept = data[:room]
                target.write(kept)
                target.flush()
                counts[index] += len(kept)
                if len(kept) != len(data):
                    truncated[index] = True
                    exceeded.set()

        threads = [threading.Thread(target=pump, args=(process.stdout, stdout, 0)),
                   threading.Thread(target=pump, args=(process.stderr, stderr, 1))]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + wall
        reason = "completed"
        while process.poll() is None:
            try:
                size = output_bytes(output)
            except (OSError, ValueError) as error:
                if first_output_error is None:
                    first_output_error = safe_scan_detail(error, "live")
                reason = "unsafe-guest-output"
                break
            if size > output_cap:
                reason = "guest-output-cap"
                break
            if exceeded.is_set():
                reason = "host-stream-cap"
                break
            if time.monotonic() >= deadline:
                reason = "wall-timeout"
                break
            time.sleep(0.5)
        if reason != "completed" and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            raw = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            raw = process.wait(timeout=15)
            reason = reason if reason != "completed" else "client-drain-timeout"
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            reason = "stream-drain-timeout"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for thread in threads:
                thread.join(timeout=2)
        if exceeded.is_set():
            reason = "host-stream-cap"
        absent = group_absent(process.pid) and not any(thread.is_alive() for thread in threads)
        if not absent:
            reason = "host-group-live"
        if not threads[0].is_alive():
            process.stdout.close()
        if not threads[1].is_alive():
            process.stderr.close()
    (root / "docker.raw-exit").write_text(f"{raw}\n")
    try:
        measured = output_bytes(output)
        final_size = str(measured)
        if measured > output_cap:
            reason = "guest-output-cap"
    except (OSError, ValueError) as error:
        if first_output_error is None:
            first_output_error = safe_scan_detail(error, "final")
        final_size = "untrusted"
        reason = "unsafe-guest-output"
    if first_output_error is None:
        first_output_error = ("none",) * 6
    error_phase, error_class, error_code, error_path, error_message, error_errno = (
        first_output_error)
    (root / "docker.monitor.tsv").write_text(
        f"reason\t{reason}\nstdout_kept_bytes\t{counts[0]}\n"
        f"stderr_kept_bytes\t{counts[1]}\n"
        f"stdout_truncated\t{str(truncated[0]).lower()}\n"
        f"stderr_truncated\t{str(truncated[1]).lower()}\n"
        f"host_group_absent\t{str(absent).lower()}\n"
        f"guest_output_bytes\t{final_size}\n"
        f"first_output_error_phase\t{error_phase}\n"
        f"first_output_error_class\t{error_class}\n"
        f"first_output_error_code\t{error_code}\n"
        f"first_output_error_path\t{error_path}\n"
        f"first_output_error_message\t{error_message}\n"
        f"first_output_error_errno\t{error_errno}\n")
    status = 0 if raw == 0 and reason == "completed" else 74
    (root / "docker.exit").write_text(f"{status}\n")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output-cap", type=int, required=True)
    parser.add_argument("--wall", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        status = run(args.run, args.output, args.cap, args.output_cap,
                     args.wall, command)
    except Exception as error:
        (args.run / "docker.monitor-error.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise
    raise SystemExit(status)


if __name__ == "__main__":
    main()
