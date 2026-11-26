#!/usr/bin/env python3
"""Verify a downloaded CLI3 gate artifact before running its fixed profile."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile


MANIFESTS = (
    "source_closure", "native_source_closure", "public_strings",
    "public_exports", "undefined_symbols",
)
MAX_OUTER_SEAL_BYTES = 1024
MAX_METADATA_BYTES = 1024 * 1024
FILES = {
    "eshkol-transformer", "SHA256SUMS",
    *(f"metadata/{name}.txt" for name in MANIFESTS),
    *(f"metadata/{name}" for name in (
        "tested-binary.sha256", "production-generate.pass",
        "cross-command-chain.pass", "pretrain.json", "greedy-first.json",
        "categorical-first.json", "eshkol.lock", "toolchain-provenance.tsv",
        "elf.txt", "dependencies.txt", "identity.tsv")),
    *(f"smoke/{name}" for name in (
        "config.json", "byte.t1", "tokenizer.json", "tokenizer.stderr",
        "version.expected", "version.stdout", "version.stderr")),
}


def fail(message: str) -> None:
    raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_stream(stream) -> str:
    hasher = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        hasher.update(chunk)
    return hasher.hexdigest()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


class ArtifactMember(tarfile.TarInfo):
    @classmethod
    def fromtarfile(cls, archive):
        header = archive.fileobj.read(tarfile.BLOCKSIZE)
        member = cls.frombuf(header, archive.encoding, archive.errors)
        # Check the raw USTAR typeflag before tarfile can consume an extension's body.
        if header[156:157] not in (tarfile.REGTYPE, tarfile.DIRTYPE):
            fail(f"unsafe or duplicate archive member: {member.name}")
        member.offset = archive.fileobj.tell() - tarfile.BLOCKSIZE
        return member._proc_member(archive)


def receipt(data: bytes, filename: str) -> str:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\n]+)\n", data)
    if not match or match.group(2).decode("ascii", "replace") != filename:
        fail(f"invalid SHA-256 receipt for {filename}")
    return match.group(1).decode("ascii")


def git(source: Path, *args: str, input_text: str | None = None) -> str:
    # Also disable transports for older Git versions without GIT_NO_LAZY_FETCH.
    environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(source), *args],
        input=input_text, text=True, stderr=subprocess.DEVNULL, env=environment,
    ).strip()


def verify_source(source: Path, files: dict[str, bytes]) -> None:
    identity = re.fullmatch(
        r"format\teshkol-cli3-binary-v1\nhead\t([0-9a-f]{40})\n"
        r"tree\t([0-9a-f]{40})\nos\tLinux\narch\tx86_64\n",
        files["metadata/identity.tsv"].decode("ascii", "strict"),
    )
    if not identity:
        fail("artifact source or platform identity differs from expected checkout")
    build_commit, build_tree = identity.groups()
    try:
        checkout_tree = git(source, "rev-parse", "HEAD^{tree}")
        dirty = git(source, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            fail("expected source checkout is dirty")
        if build_tree != checkout_tree:
            fail("artifact source or platform identity differs from expected checkout")
        object_type = git(source, "cat-file", "--batch-check=%(objecttype)",
                          input_text=f"{build_commit}\n")
        if object_type != f"{build_commit} missing" and (
                object_type != "commit" or
                git(source, "rev-parse", f"{build_commit}^{{tree}}") != build_tree):
            fail("artifact build commit does not identify the recorded tree")
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"expected source is not a readable Git checkout: {exc}")
    for name in MANIFESTS:
        source_file = source / "native" / f"cli3_generate_{name}.txt"
        if source_file.read_bytes() != files[f"metadata/{name}.txt"]:
            fail(f"artifact source manifest differs: {name}")
    if (source / "toolchain/eshkol.lock").read_bytes() != files["metadata/eshkol.lock"]:
        fail("artifact toolchain lock differs")


def verify_files(files: dict[str, bytes], hashes: dict[str, str]) -> None:
    lines = files["SHA256SUMS"].splitlines(keepends=True)
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(rb"([0-9a-f]{64})  \./([^\n]+)\n", line)
        if not match:
            fail("malformed inner SHA256SUMS")
        name = match.group(2).decode("ascii", "strict")
        if name in seen or name not in FILES - {"SHA256SUMS"}:
            fail("duplicate or unexpected inner SHA256SUMS entry")
        seen.add(name)
        if hashes[name] != match.group(1).decode("ascii"):
            fail(f"inner SHA-256 mismatch: {name}")
    if seen != FILES - {"SHA256SUMS"}:
        fail("incomplete inner SHA256SUMS")
    if receipt(files["metadata/tested-binary.sha256"], "eshkol-transformer") != hashes["eshkol-transformer"]:
        fail("tested executable receipt differs")
    for name, expected in (
        ("production-generate", b"CLI3-PRODUCTION-GENERATE-PASS\n"),
        ("cross-command-chain", b"CLI3-CROSS-COMMAND-CHAIN-PASS\n"),
    ):
        if files[f"metadata/{name}.pass"] != expected:
            fail(f"invalid {name} gate receipt")
    if files["smoke/version.expected"] != b"eshkol-transformer 0.0.1\n" or files["smoke/version.stdout"] != files["smoke/version.expected"] or files["smoke/version.stderr"] or files["smoke/tokenizer.stderr"]:
        fail("artifact smoke receipt differs")


def verify_archive(directory: Path, source: Path) -> tuple[Path, list[tarfile.TarInfo], dict[str, str]]:
    entries = list(directory.iterdir())
    if len(entries) != 2 or any(not p.is_file() or p.is_symlink() for p in entries):
        fail("artifact directory must contain exactly the two regular download files")
    archives = [p for p in entries if p.name.endswith(".tar")]
    if len(archives) != 1:
        fail("artifact directory must contain one tar archive")
    archive = archives[0]
    root = archive.name[:-4]
    if root in ("", ".", ".."):
        fail("archive root name is invalid")
    seal = directory / f"{archive.name}.sha256"
    if seal not in entries:
        fail("outer archive SHA-256 mismatch")
    with seal.open("rb") as stream:
        outer_seal = stream.read(MAX_OUTER_SEAL_BYTES + 1)
    if len(outer_seal) > MAX_OUTER_SEAL_BYTES:
        fail("outer SHA-256 receipt exceeds size limit")
    outer_hash = receipt(outer_seal, archive.name)
    directories = {root, f"{root}/metadata", f"{root}/smoke"}
    file_names = {f"{root}/{name}": name for name in FILES}
    expected = directories | file_names.keys()
    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    members: list[tarfile.TarInfo] = []
    with archive.open("rb") as archive_stream:
        if digest_stream(archive_stream) != outer_hash:
            fail("outer archive SHA-256 mismatch")
        archive_stream.seek(0)
        with tarfile.open(fileobj=archive_stream, mode="r:", tarinfo=ArtifactMember) as tar:
            seen: set[str] = set()
            for member in tar:
                if member.name not in expected or member.name in seen:
                    fail(f"unsafe or duplicate archive member: {member.name}")
                seen.add(member.name)
                members.append(member)
                if member.type == tarfile.DIRTYPE:
                    if member.name not in directories or member.size != 0:
                        fail(f"archive member type or size differs: {member.name}")
                    continue
                if member.name not in file_names:
                    fail(f"archive member type or size differs: {member.name}")
                name = file_names[member.name]
                binary = name == "eshkol-transformer"
                if member.size < 0 or (not binary and member.size > MAX_METADATA_BYTES):
                    fail(f"archive member exceeds size limit: {member.name}")
                if (member.mode & 0o111) != (0o111 if binary else 0):
                    fail("archive executable modes differ")
                with tar.extractfile(member) as stream:
                    if binary:
                        hashes[name] = digest_stream(stream)
                    else:
                        files[name] = stream.read()
                        hashes[name] = digest(files[name])
            if seen != expected:
                fail("archive member inventory differs")
    verify_files(files, hashes)
    verify_source(source, files)
    return archive, members, hashes


def supported_host() -> bool:
    os_release = Path("/etc/os-release").read_text()
    return (re.search(r'^ID=ubuntu$', os_release, re.M) is not None and
            re.search(r'^VERSION_ID="?22\.04"?$', os_release, re.M) is not None and
            platform.machine() == "x86_64" and sys.platform == "linux")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("expected_source_checkout", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="verify seals and source without executing the binary")
    args = parser.parse_args()
    archive, members, hashes = verify_archive(args.artifact_directory, args.expected_source_checkout)
    print("CLI3 downloaded artifact seals and source identity: PASS", flush=True)
    if args.verify_only:
        return
    if not supported_host():
        fail("execution requires Ubuntu 22.04 x86-64")
    # Every member was checked before extraction. Write into a fresh private
    # directory; only regular files and directories are admitted.
    with tempfile.TemporaryDirectory(prefix="cli3-downloaded-") as temporary:
        base = Path(temporary)
        with tarfile.open(archive, "r:", tarinfo=ArtifactMember) as tar:
            for member in members:
                target = base / member.name
                if member.type == tarfile.DIRTYPE:
                    target.mkdir(mode=0o700)
                else:
                    with tar.extractfile(member) as src, target.open("xb") as dst:
                        while chunk := src.read(1024 * 1024):
                            dst.write(chunk)
                    target.chmod(0o755 if member.name.endswith("/eshkol-transformer") else 0o600)
        root = base / archive.name[:-4]
        for name, expected in hashes.items():
            if digest_file(root / name) != expected:
                fail(f"extracted artifact changed before execution: {name}")
        binary = root / "eshkol-transformer"
        runner = Path(__file__).with_name("test-cli3-downloaded-fixture.sh")
        subprocess.run(["/usr/bin/bash", str(runner), str(binary)], check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, UnicodeError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print(f"CLI3 downloaded artifact: {exc}", file=sys.stderr)
        sys.exit(1)
