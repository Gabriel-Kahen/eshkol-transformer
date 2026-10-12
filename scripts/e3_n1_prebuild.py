"""N1 OpenSSL SOURCE/PREP evidence. No build or production gate entry point."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile


SCHEMA = "eshkol-e3-n1-openssl-prebuild-input-v1"
INVENTORY_SCHEMA = "eshkol-e3-n1-openssl-tar-inventory-v1"
PREPARED_SCHEMA = "eshkol-e3-n1-openssl-prepared-v1"
BASE_COMMIT = "39f3b41ba4ebe276666af82710b3750096925943"
BASE_TREE = "eb3d877d4abf0fd0968deb968ab0b7ad10b3941a"
SOURCE_SHA256 = "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a"
PRIMARY_FINGERPRINT = "B146647E45A7B33947AB226B2A2C87D161692D40"
TOP = "openssl-3.5.9"
ARCHIVE_URL = "https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz"
SIGNATURE_URL = ARCHIVE_URL + ".asc"
CERTIFICATE_URL = "https://www.openssl-library.org/source/pubkeys.asc"
MAX_MEMBERS = 100000
MAX_EXPANDED = 4 * 1024**3
MAX_INPUT = 65536
FILE_FIELDS = {"role", "path", "kind", "mode", "owner_uid", "owner_gid",
               "symlink_chain", "ancestor_chain", "size", "sha256"}


class Hold(ValueError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path, role):
    path = Path(path)
    if not path.is_absolute():
        raise Hold("evidence-path-not-absolute")
    chain = []
    current = Path(path.anchor)
    for part in (None, *path.parts[1:]):
        if part is not None:
            current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise Hold("evidence-symlink")
        chain.append([str(current), info.st_dev, info.st_ino, info.st_mode,
                      info.st_uid, info.st_gid])
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise Hold("evidence-kind")
    return {"role": role, "path": str(path), "kind": "regular",
            "mode": stat.S_IMODE(info.st_mode), "owner_uid": info.st_uid,
            "owner_gid": info.st_gid, "symlink_chain": [], "ancestor_chain": chain[:-1],
            "size": info.st_size, "sha256": file_hash(path)}


def private_evidence_directory(parent):
    parent = Path(parent)
    if not parent.is_absolute():
        raise Hold("output-path")
    current = Path(parent.anchor)
    for part in (None, *parent.parts[1:]):
        if part is not None:
            current /= part
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise Hold("output-ancestor-kind")
    pinned = os.lstat(parent)
    if stat.S_IMODE(pinned.st_mode) != 0o700 or pinned.st_uid != os.geteuid():
        raise Hold("output-directory-custody")
    return pinned


def publish_new(path, data):
    """No-replace, durable write in one pinned private evidence directory."""
    path = Path(path)
    if not path.is_absolute() or path.name in ("", ".", ".."):
        raise Hold("output-path")
    parent = path.parent
    pinned = private_evidence_directory(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if not os.path.samestat(pinned, os.fstat(parent_fd)):
            raise Hold("output-directory-drift")
        try:
            fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise Hold("output-preexisting") from exc
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise Hold("output-short-write")
                view = view[written:]
            os.fsync(fd)
            created = os.fstat(fd)
        finally:
            os.close(fd)
        reader = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent_fd)
        try:
            if not os.path.samestat(created, os.fstat(reader)) or os.read(reader, len(data) + 1) != data:
                raise Hold("output-readback")
        finally:
            os.close(reader)
        os.fsync(parent_fd)
        if not os.path.samestat(pinned, os.lstat(parent)):
            raise Hold("output-directory-drift")
    finally:
        os.close(parent_fd)


def private_snapshot(source_path, directory, expected_hash=None):
    """Copy once into a 0700 evidence directory; verify source and copy identities."""
    source_path = Path(source_path)
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=True)
    if stat.S_IMODE(os.lstat(directory).st_mode) != 0o700 or os.lstat(directory).st_uid != os.geteuid():
        raise Hold("snapshot-directory-custody")
    target = directory / source_path.name
    before = file_entry(source_path, "acquisition-source")
    if expected_hash is not None and before["sha256"] != expected_hash:
        raise Hold("source-sha256")
    if not target.exists():
        source_fd = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600)
        try:
            with os.fdopen(source_fd, "rb", closefd=False) as source, os.fdopen(target_fd, "wb", closefd=False) as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(target_fd)
        finally:
            os.close(source_fd)
            os.close(target_fd)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    after = file_entry(source_path, "acquisition-source")
    snapshot = file_entry(target, "private-snapshot")
    if before != after or snapshot["sha256"] != before["sha256"] or snapshot["size"] != before["size"]:
        raise Hold("snapshot-source-drift")
    if stat.S_IMODE(os.lstat(target).st_mode) != 0o600:
        raise Hold("snapshot-file-permissions")
    return target


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def parse_canonical(data, *, limit=MAX_INPUT):
    if len(data) > limit:
        raise Hold("record-too-large")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise Hold("duplicate-json-key")
            out[key] = value
        return out
    try:
        value = json.loads(data.decode("ascii"), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(Hold("json-constant")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Hold("invalid-json") from exc
    if canonical(value) != data:
        raise Hold("noncanonical-json")
    return value


def _octal(field):
    value = field.rstrip(b"\0 ").lstrip(b" ")
    if not value or not re.fullmatch(rb"[0-7]+", value):
        raise Hold("tar-number")
    return int(value, 8)


def _name(raw):
    try:
        return raw.split(b"\0", 1)[0].decode("ascii")
    except UnicodeError as exc:
        raise Hold("tar-nonascii-path") from exc


def _safe_path(name, *, directory=False):
    normalized = name[:-1] if directory and name.endswith("/") else name
    parts = normalized.split("/")
    if (not normalized or normalized.startswith("/") or "\\" in normalized or
            any(part in ("", ".", "..") for part in parts) or parts[0] != TOP):
        raise Hold("tar-unsafe-path")
    return normalized


def _pax(data):
    result = {}
    cursor = 0
    while cursor < len(data):
        space = data.find(b" ", cursor)
        if space < 0 or not re.fullmatch(rb"[1-9][0-9]*", data[cursor:space]):
            raise Hold("tar-pax-length")
        length = int(data[cursor:space])
        row = data[cursor:cursor + length]
        if len(row) != length or not row.endswith(b"\n") or b"=" not in row:
            raise Hold("tar-pax-record")
        key, value = bytes(row[space - cursor + 1:-1]).split(b"=", 1)
        if key in result:
            raise Hold("tar-duplicate-pax")
        if key != b"comment" or not re.fullmatch(rb"[0-9a-f]{40}", value):
            raise Hold("tar-unreviewed-pax")
        result[key] = value.decode("ascii")
        cursor += length
    return {key.decode("ascii"): value for key, value in result.items()}


def scan_archive(path):
    """Stream raw tar headers and produce ordered members plus effective tree."""
    members = []
    seen = set()
    global_pax = None
    expanded = 0
    with gzip.open(path, "rb") as source:
        while True:
            header = source.read(512)
            if len(header) != 512:
                raise Hold("tar-truncated-header")
            if header == bytes(512):
                if source.read(512) != bytes(512):
                    raise Hold("tar-trailing-data")
                trailing = source.read(1024 * 1024 + 1)
                if len(trailing) > 1024 * 1024 or any(trailing):
                    raise Hold("tar-trailing-data")
                break
            checksum = _octal(header[148:156])
            if sum(header[:148] + b" " * 8 + header[156:]) != checksum:
                raise Hold("tar-checksum")
            size = _octal(header[124:136])
            mode = _octal(header[100:108])
            uid = _octal(header[108:116])
            gid = _octal(header[116:124])
            kind = header[156:157]
            prefix = _name(header[345:500])
            name = _name(header[:100])
            if prefix:
                name = prefix + "/" + name
            if size > MAX_EXPANDED or expanded + size > MAX_EXPANDED:
                raise Hold("tar-expanded-cap")
            expanded += size
            digest = hashlib.sha256()
            payload = bytearray() if kind == b"g" else None
            remaining = size
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise Hold("tar-truncated-member")
                digest.update(chunk)
                if payload is not None:
                    if size > 4096:
                        raise Hold("tar-pax-cap")
                    payload.extend(chunk)
                remaining -= len(chunk)
            padding = (512 - size % 512) % 512
            if source.read(padding) != bytes(padding):
                raise Hold("tar-padding")
            if kind == b"g":
                if global_pax is not None:
                    raise Hold("tar-repeated-global-pax")
                global_pax = _pax(payload)
                continue
            if kind not in (b"0", b"\0", b"5"):
                raise Hold("tar-link-or-unsupported-kind")
            directory = kind == b"5"
            if directory and size:
                raise Hold("tar-directory-data")
            normalized = _safe_path(name, directory=directory)
            if normalized in seen:
                raise Hold("tar-duplicate-effective-path")
            seen.add(normalized)
            if len(seen) > MAX_MEMBERS:
                raise Hold("tar-member-cap")
            members.append({"path": normalized, "kind": "dir" if directory else "file",
                            "mode": mode, "uid": uid, "gid": gid, "size": size,
                            "link_target": None, "sha256": None if directory else digest.hexdigest()})
    if global_pax is None or not members or members[0]["path"] != TOP:
        raise Hold("tar-root-or-pax")
    for row in members:
        parent = PurePosixPath(row["path"]).parent
        if str(parent) != "." and str(parent) not in seen:
            raise Hold("tar-implicit-parent")
    return {"schema": INVENTORY_SCHEMA, "global_pax": global_pax,
            "members": members, "effective_tree": sorted(members, key=lambda row: row["path"])}


def verify_signature(archive, signature, certificate, gpg="gpg"):
    """Verify detached bytes using only the published certificate and fixed primary."""
    with tempfile.TemporaryDirectory(prefix="e3-n1-gpg-") as home:
        os.chmod(home, 0o700)
        base = [gpg, "--homedir", home, "--batch", "--no-tty"]
        imported = subprocess.run(base + ["--import", str(certificate)], capture_output=True)
        if imported.returncode:
            raise Hold("certificate-import")
        listing = subprocess.run(base + ["--with-colons", "--fingerprint",
                                         PRIMARY_FINGERPRINT], capture_output=True)
        lines = listing.stdout.decode("ascii", "strict").splitlines()
        if listing.returncode or not any(line.startswith("fpr:::::::::" + PRIMARY_FINGERPRINT + ":")
                                             for line in lines):
            raise Hold("trusted-primary-fingerprint")
        verified = subprocess.run(base + ["--status-fd", "1", "--verify",
                                          str(signature), str(archive)], capture_output=True)
        statuses = verified.stdout.decode("utf-8", "replace").splitlines()
        valid = [line.split() for line in statuses if line.startswith("[GNUPG:] VALIDSIG ")]
        if (verified.returncode or len(valid) != 1 or len(valid[0]) < 12 or
                valid[0][-1] != PRIMARY_FINGERPRINT or
                not any(line.startswith("[GNUPG:] GOODSIG ") for line in statuses)):
            raise Hold("detached-signature")
        return {"primary_fingerprint": PRIMARY_FINGERPRINT,
                "signing_subkey_fingerprint": valid[0][2],
                "gpg_path": str(Path(gpg).resolve()),
                "gpg_sha256": file_hash(Path(gpg).resolve()),
                "import_argv": ["--import", str(certificate)],
                "verify_argv": ["--status-fd", "1", "--verify", str(signature), str(archive)],
                "result": "VALIDSIG"}


def observed_source(archive, signature, certificate, inventory_path, *, gpg="/usr/bin/gpg",
                    after_verify=None):
    archive, signature, certificate = map(Path, (archive, signature, certificate))
    private_evidence_directory(Path(inventory_path).absolute().parent)
    snapshot_dir = Path(inventory_path).absolute().parent / "source-snapshot"
    archive = private_snapshot(archive, snapshot_dir, SOURCE_SHA256)
    signature = private_snapshot(signature, snapshot_dir)
    certificate = private_snapshot(certificate, snapshot_dir)
    before = [file_entry(path, role) for path, role in (
        (archive, "source-archive"), (signature, "detached-signature"),
        (certificate, "trusted-certificate"))]
    proof = verify_signature(archive, signature, certificate, gpg)
    if after_verify is not None:
        after_verify(archive)
    inventory = canonical(scan_archive(archive))
    if before != [file_entry(path, role) for path, role in (
            (archive, "source-archive"), (signature, "detached-signature"),
            (certificate, "trusted-certificate"))]:
        raise Hold("source-changed-during-verification")
    publish_new(Path(inventory_path).absolute(), inventory)
    return {"archive": {"url": ARCHIVE_URL, **before[0]},
            "signature": {"url": SIGNATURE_URL, **before[1]},
            "certificate": {"url": CERTIFICATE_URL, **before[2]},
            "verification": proof,
            "inventory": {**file_entry(Path(inventory_path).absolute(), "member-inventory"),
                          "members": len(json.loads(inventory)["members"])}}


def candidate(source, lock_path):
    return {"schema": SCHEMA, "status": "HOLD", "base": {"commit": BASE_COMMIT, "tree": BASE_TREE},
            "source": source, "toolchain_lock": file_entry(Path(lock_path).absolute(), "toolchain-lock"),
            "supported_build": None, "tools": None, "existing_headers": None,
            "protected_paths": None, "configure_build_test_install_argv": None,
            "verifier_source_abi_and_argv": None, "complete_environment": None,
            "build_policy": {"target": "linux-x86_64", "link": "shared-libcrypto.so.3",
                             "libdir": "lib", "prefix": None, "openssldir": None,
                             "environment": "explicit-reviewed-map-only-no-inheritance",
                             "network": "disabled", "install_system_paths": False},
            "discovery": {"archive": "raw-header-safe-scan-v1",
                          "headers": "depfiles-plus-include-file-open-trace",
                          "elf": "readelf-plus-controlled-loader-and-process-maps",
                          "reject": "missing-duplicate-added-changed-unsafe-or-unreviewed"}}


def recheck_source(source):
    if set(source) != {"archive", "signature", "certificate", "verification", "inventory"}:
        raise Hold("source-evidence-fields")
    paths = {}
    roles = {"archive": "source-archive", "signature": "detached-signature",
             "certificate": "trusted-certificate", "inventory": "member-inventory"}
    for name in ("archive", "signature", "certificate", "inventory"):
        entry = source[name]
        if set(entry) != (FILE_FIELDS | ({"members"} if name == "inventory" else {"url"})):
            raise Hold("source-file-fields")
        if entry["role"] != roles[name]:
            raise Hold("source-role-drift")
        if name != "inventory" and entry["url"] != {
                "archive": ARCHIVE_URL, "signature": SIGNATURE_URL,
                "certificate": CERTIFICATE_URL}[name]:
            raise Hold("source-url-drift")
        path = Path(entry["path"])
        expected = {key: value for key, value in entry.items() if key not in ("url", "members")}
        if file_entry(path, expected["role"]) != expected:
            raise Hold("source-file-drift")
        paths[name] = path
    if source["archive"]["sha256"] != SOURCE_SHA256:
        raise Hold("source-sha256")
    before = [file_entry(paths[name], source[name]["role"]) for name in
              ("archive", "signature", "certificate")]
    proof = verify_signature(paths["archive"], paths["signature"], paths["certificate"],
                             source["verification"]["gpg_path"])
    if proof != source["verification"]:
        raise Hold("signature-proof-drift")
    inventory = canonical(scan_archive(paths["archive"]))
    if before != [file_entry(paths[name], source[name]["role"]) for name in
                  ("archive", "signature", "certificate")]:
        raise Hold("source-changed-during-verification")
    if (inventory != paths["inventory"].read_bytes() or
            file_entry(paths["inventory"], "member-inventory") != {
                key: value for key, value in source["inventory"].items() if key != "members"} or
            len(json.loads(inventory)["members"]) != source["inventory"]["members"]):
        raise Hold("inventory-drift")


def validate_candidate(data, expected_digest=None, *, recheck=False):
    value = parse_canonical(data)
    if expected_digest is not None and sha256(data) != expected_digest:
        raise Hold("candidate-digest-drift")
    if value.get("schema") != SCHEMA or value.get("base") != {
            "commit": BASE_COMMIT, "tree": BASE_TREE}:
        raise Hold("base-or-schema-drift")
    if value.get("source", {}).get("archive", {}).get("sha256") != SOURCE_SHA256:
        raise Hold("source-pin-drift")
    if value.get("source", {}).get("verification", {}).get("primary_fingerprint") != PRIMARY_FINGERPRINT:
        raise Hold("trust-anchor-drift")
    if value != candidate(value["source"], Path(__file__).resolve().parents[1] / "toolchain/eshkol.lock"):
        raise Hold("candidate-field-drift")
    if recheck:
        recheck_source(value["source"])
    return value


def plan_gate(candidate_bytes, reviewed_digest, nonce):
    """Read-only production plan: current SOURCE/PREP candidates cannot authorize a build."""
    value = validate_candidate(candidate_bytes, reviewed_digest, recheck=True)
    if value["status"] != "FROZEN":
        raise Hold("prebuild-input-not-frozen")
    return prepared_record(sha256(candidate_bytes), nonce, reviewed_digest)


def prepared_record(input_digest, nonce, reviewed_digest, *, base_commit=BASE_COMMIT,
                    base_tree=BASE_TREE, source_sha256=SOURCE_SHA256):
    if (not re.fullmatch(r"[0-9a-f]{64}", input_digest) or
            not re.fullmatch(r"[0-9a-f]{64}", nonce) or input_digest != reviewed_digest or
            base_commit != BASE_COMMIT or base_tree != BASE_TREE or source_sha256 != SOURCE_SHA256):
        raise Hold("prepared-pin-mismatch")
    return {"schema": PREPARED_SCHEMA, "base_commit": base_commit,
            "base_tree": base_tree, "source_sha256": source_sha256,
            "input_sha256": input_digest, "nonce": nonce,
            "authorized_next_step": "one-exact-openssl-build-attempt"}


def consume_once(record, directory, expected_chain, *, fault=None):
    """Low-level one-use primitive for synthetic tests; no production CLI uses it."""
    path = Path(directory)
    if (not isinstance(record, dict) or record != prepared_record(
            record.get("input_sha256", ""), record.get("nonce", ""),
            record.get("input_sha256", ""), base_commit=record.get("base_commit"),
            base_tree=record.get("base_tree"), source_sha256=record.get("source_sha256"))):
        raise Hold("marker-record-drift")
    chain = []
    current = Path(path.anchor)
    for part in (None, *path.parts[1:]):
        if part is not None:
            current /= part
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise Hold("marker-ancestor-kind")
        chain.append([info.st_dev, info.st_ino, info.st_mode, info.st_uid])
    if chain != expected_chain or chain[-1][3] != os.geteuid() or chain[-1][2] & 0o077:
        raise Hold("marker-ancestor-drift-or-permissions")
    directory_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for index, part in enumerate(path.parts[1:], start=1):
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                               dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
            info = os.fstat(directory_fd)
            if [info.st_dev, info.st_ino, info.st_mode, info.st_uid] != expected_chain[index]:
                raise Hold("marker-open-chain-drift")
        if fault == "before-create":
            raise Hold("injected-before-create")
        name = "spent-" + record["nonce"] + ".json"
        data = canonical(record)
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory_fd)
        except FileExistsError as exc:
            raise Hold("marker-spent") from exc
        try:
            if fault == "after-create":
                raise Hold("injected-after-create")
            if os.write(fd, data) != len(data):
                raise Hold("marker-short-write")
            if fault == "after-write":
                raise Hold("injected-after-write")
            os.fsync(fd)
        finally:
            os.close(fd)
        reader = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
        try:
            if os.read(reader, len(data) + 1) != data:
                raise Hold("marker-readback")
        finally:
            os.close(reader)
        if fault == "after-readback":
            raise Hold("injected-after-readback")
        os.fsync(directory_fd)
        if fault == "after-sync":
            raise Hold("injected-after-sync")
    finally:
        os.close(directory_fd)


def path_chain(path):
    result = []
    current = Path(Path(path).anchor)
    for part in (None, *Path(path).parts[1:]):
        if part is not None:
            current /= part
        info = os.lstat(current)
        result.append([info.st_dev, info.st_ino, info.st_mode, info.st_uid])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    if Path(args.inventory).absolute().parent != Path(args.candidate).absolute().parent:
        raise Hold("output-directory-mismatch")
    source = observed_source(args.archive, args.signature, args.certificate, args.inventory)
    output = canonical(candidate(source, Path(__file__).resolve().parents[1] / "toolchain/eshkol.lock"))
    publish_new(Path(args.candidate).absolute(), output)
    print(sha256(output), args.candidate)
    print("HOLD: supported image/tools/headers/paths/argv/env and ROOT review absent")


if __name__ == "__main__":
    main()
