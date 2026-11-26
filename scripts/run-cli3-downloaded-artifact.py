#!/usr/bin/env python3
"""Verify a downloaded CLI3 gate artifact before running its fixed profile."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
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


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def receipt(data: bytes, filename: str) -> str:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\n]+)\n", data)
    if not match or match.group(2).decode("ascii", "replace") != filename:
        fail(f"invalid SHA-256 receipt for {filename}")
    return match.group(1).decode("ascii")


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def verify_source(source: Path, files: dict[str, bytes]) -> None:
    try:
        head = git(source, "rev-parse", "HEAD")
        tree = git(source, "rev-parse", "HEAD^{tree}")
        dirty = git(source, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"expected source is not a readable Git checkout: {exc}")
    if dirty:
        fail("expected source checkout is dirty")
    identity = files["metadata/identity.tsv"].decode("ascii", "strict")
    expected = ("format\teshkol-cli3-binary-v1\n"
                f"head\t{head}\ntree\t{tree}\nos\tLinux\narch\tx86_64\n")
    if identity != expected:
        fail("artifact source or platform identity differs from expected checkout")
    for name in MANIFESTS:
        source_file = source / "native" / f"cli3_generate_{name}.txt"
        if source_file.read_bytes() != files[f"metadata/{name}.txt"]:
            fail(f"artifact source manifest differs: {name}")
    if (source / "toolchain/eshkol.lock").read_bytes() != files["metadata/eshkol.lock"]:
        fail("artifact toolchain lock differs")


def verify_files(files: dict[str, bytes], hashes: dict[str, str]) -> None:
    if set(files) != FILES:
        fail(f"artifact file inventory differs: missing={sorted(FILES-set(files))}, extra={sorted(set(files)-FILES)}")
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
    seal = directory / f"{archive.name}.sha256"
    if seal not in entries or receipt(seal.read_bytes(), archive.name) != digest_file(archive):
        fail("outer archive SHA-256 mismatch")
    root = archive.name[:-4]
    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    members: list[tarfile.TarInfo] = []
    with tarfile.open(archive, "r:") as tar:
        seen: set[str] = set()
        for member in tar:
            raw = member.name.rstrip("/")
            parts = PurePosixPath(raw).parts
            if (not raw or raw.startswith("/") or parts[0] != root or
                    any(part in (".", "..", "") for part in raw.split("/")) or
                    raw in seen or not (member.isdir() or member.isfile())):
                fail(f"unsafe or duplicate archive member: {member.name}")
            seen.add(raw)
            members.append(member)
            if member.isfile():
                stream = tar.extractfile(member)
                assert stream is not None
                name = "/".join(parts[1:])
                hasher = hashlib.sha256()
                data = bytearray()
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
                    if name != "eshkol-transformer":
                        data.extend(chunk)
                files[name] = bytes(data)
                hashes[name] = hasher.hexdigest()
        if set(seen) != {root, f"{root}/metadata", f"{root}/smoke"} | {f"{root}/{p}" for p in FILES}:
            fail("archive member inventory differs")
        if any(m.isfile() and (m.mode & 0o111) != (0o111 if m.name == f"{root}/eshkol-transformer" else 0) for m in members):
            fail("archive executable modes differ")
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
        with tarfile.open(archive, "r:") as tar:
            for member in members:
                target = base.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(mode=0o700)
                else:
                    with tar.extractfile(member) as src, target.open("xb") as dst:
                        assert src is not None
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
