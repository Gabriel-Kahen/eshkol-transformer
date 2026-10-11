#!/usr/bin/env python3
"""Inert E3 N1 OCI export acquisition. No production pin has been released.

This source reads no layer unless an exact, separately reviewed pin SHA is
installed below. It never invokes Docker, decompresses, or executes a blob.
"""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import secrets
import stat
import time
from pathlib import Path

RELEASED_PIN_SHA256 = None  # ROOT review must install an immutable pin and release.
EXPECTED = {
    "index": ("f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6", 855),
    "manifest": ("3649921cfd39ef9b2c4a54d0c6ca01cd6fc49768806297fe61e05a70648804fb", 676),
    "config": ("584fc09fcf7a84e381d8ef1f2903143b7b1debbbd5661468ed31d56863087ee2", 2326),
    "layers": [
        ("20c3783cc497b5b0df1fc5f92bd64c3d2fbb24057c88692c8fda205b6ea8a2f2", 29750251),
        ("399066089356a53b7c7046be9a8a2e938647e6a4076724bf42034912ea6ace98", 449869772),
    ],
    "diff_ids": [
        "sha256:ea16cace89338c84eb6bcb91a7efdfcae6838fff359efe951858227436486c34",
        "sha256:5ef3a0f7c37c1e91eaf25f50ed9c318abd80e417aef95d9352435bb165eeb458",
    ],
}
MAX_PIN = 64 * 1024
MAX_SMALL = 1024 * 1024
MAX_BLOB = MAX_COMBINED = 512 * 1024 * 1024
MAX_DISK = 520 * 1024 * 1024
MAX_SECONDS = 1800
MAX_RSS = 2 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024
FILE_KEYS = {"path", "resolved_path", "mode", "size", "device", "inode", "mtime_ns", "ctime_ns", "sha256"}
SOURCE_KEYS = FILE_KEYS - {"sha256"}


class Hold(Exception):
    pass


def need(value, reason):
    if not value:
        raise Hold(reason)


def canonical_json(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate-json-key")
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=unique)
    except (UnicodeError, ValueError) as error:
        raise Hold("invalid-json") from error


def digest(value):
    need(isinstance(value, str) and len(value) == 64 and
         all(ch in "0123456789abcdef" for ch in value), "invalid-sha256")
    return value


def positive(value, cap, reason):
    need(type(value) is int and 0 < value <= cap, reason)
    return value


def absolute(path):
    need(isinstance(path, str) and path.startswith("/") and "\x00" not in path and
         os.path.normpath(path) == path, "invalid-path")
    return path


def metadata(st):
    return {"mode": st.st_mode, "size": st.st_size, "device": st.st_dev,
            "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def open_directory(path):
    """Anchor a directory through no-follow opens of every component."""
    path = absolute(path)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        result = directory
        directory = None
        return result
    finally:
        if directory is not None:
            os.close(directory)


def open_regular(path):
    """Walk every component without following a symlink, including parents."""
    path = absolute(path)
    need(path != "/", "invalid-path")
    directory = open_directory(os.path.dirname(path))
    try:
        fd = os.open(os.path.basename(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        need(stat.S_ISREG(os.fstat(fd).st_mode), "nonregular-source")
        return fd
    finally:
        os.close(directory)


def check_identity(path, pin, fd=None):
    need(set(pin) in (FILE_KEYS, SOURCE_KEYS), "file-pin-keys")
    path = absolute(path)
    need(pin["path"] == path and pin["resolved_path"] == path and
         os.path.realpath(path) == path, "path-or-symlink-drift")
    st = os.lstat(path)
    need(stat.S_ISREG(st.st_mode), "nonregular-source")
    need(metadata(st) == {key: pin[key] for key in SOURCE_KEYS if key not in ("path", "resolved_path")},
         "source-identity-drift")
    if fd is not None:
        need(metadata(os.fstat(fd)) == metadata(st), "source-fd-drift")


def read_small(pin, expected, deadline, rss_cap):
    need(set(pin) == FILE_KEYS, "small-pin-keys")
    path = absolute(pin["path"])
    need(pin["size"] == expected[1] and 0 < pin["size"] <= MAX_SMALL and
         digest(pin["sha256"]) == expected[0], "small-descriptor-drift")
    check_identity(path, pin)
    fd = open_regular(path)
    try:
        check_identity(path, pin, fd)
        data = os.read(fd, pin["size"] + 1)
        sample(deadline, rss_cap)
        need(len(data) == pin["size"] and hashlib.sha256(data).hexdigest() == pin["sha256"],
             "small-byte-drift")
        check_identity(path, pin, fd)
        return data
    finally:
        os.close(fd)


def sample(deadline, rss_cap):
    need(time.monotonic() <= deadline, "time-cap")
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= rss_cap, "rss-cap")


def descriptor(obj, expected, media):
    need(isinstance(obj, dict) and obj.get("mediaType") == media and
         obj.get("digest") == "sha256:" + expected[0] and
         type(obj.get("size")) is int and obj["size"] == expected[1], "descriptor-drift")


def validate_chain(small, layers, expected):
    index, manifest, config = (parse_json(small[name]) for name in ("index", "manifest", "config"))
    need(isinstance(index, dict) and isinstance(index.get("manifests"), list), "index-shape")
    amd64 = [d for d in index["manifests"] if isinstance(d, dict) and
             d.get("platform") == {"architecture": "amd64", "os": "linux"}]
    need(len(amd64) == 1, "amd64-manifest-count")
    descriptor(amd64[0], expected["manifest"], "application/vnd.oci.image.manifest.v1+json")
    need(isinstance(manifest, dict), "manifest-shape")
    descriptor(manifest.get("config"), expected["config"], "application/vnd.oci.image.config.v1+json")
    need(manifest.get("layers") == layers and len(layers) == 2, "layer-order-drift")
    for layer, pair in zip(layers, expected["layers"]):
        descriptor(layer, pair, "application/vnd.oci.image.layer.v1.tar+gzip")
    need(isinstance(config, dict) and config.get("architecture") == "amd64" and
         config.get("os") == "linux" and isinstance(config.get("rootfs"), dict) and
         config["rootfs"].get("diff_ids") == expected["diff_ids"], "config-platform-or-diffids")


def validate_pin(pin, expected, deadline):
    need(isinstance(pin, dict) and set(pin) == {"schema", "small", "layers", "sources", "destination", "limits"}
         and pin["schema"] == 1, "pin-schema")
    limits = pin["limits"]
    need(isinstance(limits, dict) and set(limits) == {"seconds", "max_rss_bytes", "max_disk_bytes",
                                                "max_blob_bytes", "max_combined_bytes"}, "limit-schema")
    positive(limits["seconds"], MAX_SECONDS, "time-limit")
    positive(limits["max_rss_bytes"], MAX_RSS, "rss-limit")
    positive(limits["max_disk_bytes"], MAX_DISK, "disk-limit")
    positive(limits["max_blob_bytes"], MAX_BLOB, "blob-limit")
    positive(limits["max_combined_bytes"], MAX_COMBINED, "combined-limit")
    small_pins = pin["small"]
    need(isinstance(small_pins, dict) and set(small_pins) == {"index", "manifest", "config"}, "small-schema")
    need(isinstance(pin["layers"], list) and len(pin["layers"]) == 2 and
         isinstance(pin["sources"], list) and len(pin["sources"]) == 2, "layer-schema")
    sizes = [pair[1] for pair in expected["layers"]]
    need(all(0 < size <= limits["max_blob_bytes"] for size in sizes) and
         sum(sizes) <= limits["max_combined_bytes"], "descriptor-byte-cap")
    need(sum(sizes) + sum(pair[1] for pair in (expected["index"], expected["manifest"], expected["config"])) + 65536
         <= limits["max_disk_bytes"], "descriptor-disk-cap")
    dest = pin["destination"]
    need(isinstance(dest, dict) and set(dest) == {"path", "parent", "parent_mode", "parent_device",
                                                   "parent_inode", "free_bytes"}, "destination-schema")
    target = absolute(dest["path"])
    parent = absolute(dest["parent"])
    need(os.path.dirname(target) == parent and os.path.realpath(parent) == parent and
         not os.path.lexists(target), "destination-drift")
    st = os.lstat(parent)
    need(stat.S_ISDIR(st.st_mode) and st.st_mode == dest["parent_mode"] and
         st.st_dev == dest["parent_device"] and st.st_ino == dest["parent_inode"], "destination-parent-drift")
    free = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
    need(type(dest["free_bytes"]) is int and dest["free_bytes"] >= limits["max_disk_bytes"] and
         free >= limits["max_disk_bytes"], "destination-free-cap")
    small = {name: read_small(small_pins[name], expected[name], deadline, limits["max_rss_bytes"])
             for name in ("index", "manifest", "config")}
    validate_chain(small, pin["layers"], expected)
    for source in pin["sources"]:
        need(isinstance(source, dict) and set(source) == SOURCE_KEYS, "source-pin-keys")
        check_identity(source["path"], source)
    return small


def write_new(path, data, hashes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    h = hashlib.sha256()
    count = 0
    hashes[str(path)] = (h.hexdigest(), count)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            need(written > 0, "write-stalled")
            h.update(view[:written])
            count += written
            hashes[str(path)] = (h.hexdigest(), count)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_layer(source_pin, descriptor_pin, temp_path, limits, deadline, hashes, used):
    path = source_pin["path"]
    need(source_pin["size"] == descriptor_pin["size"], "source-descriptor-size")
    check_identity(path, source_pin)
    source = open_regular(path)
    target = None
    h = hashlib.sha256()
    count = 0
    written_total = 0
    try:
        check_identity(path, source_pin, source)
        target = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        hashes[str(temp_path)] = (h.hexdigest(), 0)
        while True:
            sample(deadline, limits["max_rss_bytes"])
            block = os.read(source, min(CHUNK, descriptor_pin["size"] - count + 1))
            sample(deadline, limits["max_rss_bytes"])
            if not block:
                break
            count += len(block)
            need(count <= descriptor_pin["size"] and count <= limits["max_blob_bytes"] and
                 used + count <= limits["max_combined_bytes"] and
                 used + count <= limits["max_disk_bytes"], "copy-byte-cap")
            view = memoryview(block)
            while view:
                written = os.write(target, view)
                need(written > 0, "write-stalled")
                h.update(view[:written])
                written_total += written
                hashes[str(temp_path)] = (h.hexdigest(), written_total)
                view = view[written:]
                sample(deadline, limits["max_rss_bytes"])
        os.fsync(target)
        check_identity(path, source_pin, source)
        need(count == descriptor_pin["size"] and h.hexdigest() == descriptor_pin["digest"][7:],
             "layer-byte-drift")
        sample(deadline, limits["max_rss_bytes"])
        hashes[str(temp_path)] = (h.hexdigest(), count)
        return count
    finally:
        if target is not None:
            os.close(target)
        os.close(source)


def seal(stage, stage_fd, hashes, status, reason, pin_sha):
    record = {"schema": 1, "status": status, "reason": reason, "pin_sha256": pin_sha}
    name = "ACQUISITION.json" if status == "VERIFIED" else "HOLD.json"
    write_new(stage / name, canonical_json(record), hashes)
    rows = [f"{digest}  {Path(path).relative_to(stage)}\n" for path, (digest, _) in sorted(hashes.items())]
    write_new(stage / "SHA256SUMS-ACQUISITION", "".join(rows).encode(), {})
    os.fsync(stage_fd)


def rename_noreplace(parent_fd, source_name, target_name):
    need("/" not in source_name and "/" not in target_name and source_name and target_name,
         "rename-name")
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    result = function(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(target_name), 1)
    if result != 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()), target_name)


def check_named_parent(dest, parent_fd, target_name=None, stage_identity=None):
    """Require the requested path still to name the anchored parent and target."""
    parent = dest["parent"]
    need(os.path.realpath(parent) == parent, "destination-parent-drift")
    named = os.lstat(parent)
    anchored = os.fstat(parent_fd)
    need(stat.S_ISDIR(named.st_mode) and named.st_mode == dest["parent_mode"] and
         named.st_dev == dest["parent_device"] and named.st_ino == dest["parent_inode"] and
         (named.st_dev, named.st_ino) == (anchored.st_dev, anchored.st_ino),
         "destination-parent-drift")
    if target_name is not None:
        visible = os.lstat(dest["path"])
        held = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        need(stat.S_ISDIR(visible.st_mode) and
             (visible.st_dev, visible.st_ino) == (held.st_dev, held.st_ino) == stage_identity,
             "destination-target-drift")
        final_parent = os.lstat(parent)
        need(os.path.realpath(parent) == parent and stat.S_ISDIR(final_parent.st_mode) and
             (final_parent.st_dev, final_parent.st_ino) == (anchored.st_dev, anchored.st_ino),
             "destination-parent-drift")


def create_stage(parent_fd):
    for _ in range(8):
        name = ".e3-n1-HOLD-" + secrets.token_hex(12)
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            return name
        except FileExistsError:
            continue
    raise Hold("stage-name-collision")


def run(pin_path):
    need(RELEASED_PIN_SHA256 is not None, "release-pin-absent")
    digest(RELEASED_PIN_SHA256)
    pin_path = absolute(pin_path)
    fd = open_regular(pin_path)
    try:
        st = os.fstat(fd)
        need(0 < st.st_size <= MAX_PIN, "pin-byte-cap")
        raw = os.read(fd, MAX_PIN + 1)
        need(len(raw) == st.st_size and metadata(os.fstat(fd)) == metadata(st), "pin-byte-drift")
    finally:
        os.close(fd)
    pin_sha = hashlib.sha256(raw).hexdigest()
    need(pin_sha == RELEASED_PIN_SHA256, "release-pin-drift")
    pin = parse_json(raw)
    limits = pin.get("limits", {}) if isinstance(pin, dict) else {}
    need(isinstance(limits, dict), "limit-schema")
    seconds = positive(limits.get("seconds"), MAX_SECONDS, "time-limit")
    deadline = time.monotonic() + seconds
    try:
        small = validate_pin(pin, EXPECTED, deadline)
    except (TypeError, KeyError, IndexError, ValueError) as error:
        raise Hold("invalid-pin") from error
    dest = pin["destination"]
    parent_fd = open_directory(dest["parent"])
    try:
        check_named_parent(dest, parent_fd)
        stage_name = create_stage(parent_fd)
        try:
            stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=parent_fd)
        except Exception:
            try:
                os.rmdir(stage_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as cleanup_error:
                raise Hold("stage-acquisition-cleanup-failed") from cleanup_error
            raise
    except Exception:
        os.close(parent_fd)
        raise
    # This fd continues to identify the same directory if its name is moved.
    stage = Path(f"/proc/self/fd/{stage_fd}")
    stage_st = os.fstat(stage_fd)
    stage_identity = (stage_st.st_dev, stage_st.st_ino)
    hashes = {}
    try:
        reserve = stage / ".hold-reserve"
        reserve_fd = os.open(reserve, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.posix_fallocate(reserve_fd, 0, 16 * 1024)
        finally:
            os.close(reserve_fd)
        blob_dir = stage / "blobs" / "sha256"
        blob_dir.mkdir(parents=True, mode=0o700)
        used = 0
        parts = []
        for source, layer in zip(pin["sources"], pin["layers"]):
            temp = blob_dir / (layer["digest"][7:] + ".part")
            count = copy_layer(source, layer, temp, limits, deadline, hashes, used)
            used += count
            parts.append(temp)
        sample(deadline, limits["max_rss_bytes"])
        for temp in parts:
            final = temp.with_suffix("")
            os.rename(temp, final)
            hashes[str(final)] = hashes.pop(str(temp))
        for name, pair in (("manifest", EXPECTED["manifest"]), ("config", EXPECTED["config"])):
            output = blob_dir / pair[0]
            write_new(output, small[name], hashes)
        write_new(stage / "index.json", small["index"], hashes)
        write_new(stage / "oci-layout", b'{"imageLayoutVersion":"1.0.0"}\n', hashes)
        sample(deadline, limits["max_rss_bytes"])
        check_named_parent(dest, parent_fd)
        need(not os.path.lexists(dest["path"]) and
             not os.path.lexists(os.path.join(f"/proc/self/fd/{parent_fd}",
                                                os.path.basename(dest["path"]))),
             "destination-drift")
        reserve.unlink()
        seal(stage, stage_fd, hashes, "VERIFIED", None, pin_sha)
        sample(deadline, limits["max_rss_bytes"])
        os.fsync(parent_fd)
        target_name = os.path.basename(dest["path"])
        rename_noreplace(parent_fd, stage_name, target_name)
        try:
            os.fsync(parent_fd)
            check_named_parent(dest, parent_fd, target_name, stage_identity)
        except Exception as error:
            try:
                rename_noreplace(parent_fd, target_name, stage_name)
            except Exception:
                # The retained stage fd still reaches the published directory,
                # even if another actor moved it to a different name.
                pass
            raise Hold("destination-parent-drift") from error
        return dest["path"]
    except Exception as error:
        current = os.fstat(stage_fd)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != stage_identity:
            raise Hold("publication-retraction-failed") from error
        reserve = stage / ".hold-reserve"
        if reserve.exists():
            reserve.unlink()
        if (stage / "SHA256SUMS-ACQUISITION").exists():
            (stage / "SHA256SUMS-ACQUISITION").unlink()
        if (stage / "ACQUISITION.json").exists():
            (stage / "ACQUISITION.json").unlink()
            hashes.pop(str(stage / "ACQUISITION.json"), None)
        reason = str(error) if isinstance(error, Hold) else type(error).__name__
        seal(stage, stage_fd, hashes, "HOLD", reason[:120], pin_sha)
        raise
    finally:
        os.close(stage_fd)
        os.close(parent_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True, help="exact ROOT-reviewed metadata-only pin JSON")
    args = parser.parse_args()
    try:
        result = run(args.pin)
    except (Hold, OSError) as error:
        print(canonical_json({"status": "HOLD", "reason": str(error)}).decode(), end="")
        return 1
    print(canonical_json({"status": "VERIFIED", "export": result}).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
