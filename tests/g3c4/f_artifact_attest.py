"""Test-only admission for a copied, observed run-F compiler artifact.

This never turns the historical failed F gate into a passing gate. A ROOT-owned
launcher must pin the accepted offline result, image and live resource graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


COMPILER_HEAD = "6803dbe24fb65eaad2901767bf010787517946ef"
COMPILER_TREE = "539d22b1b6a89c290aa13c91263982332b01b87b"
TRACKED_LOCK_SHA = "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe"
CANDIDATE_LOCK_SHA = "b798822b18535405092b201248d8e37682d5ddb9555b6080ee32131659bfbea5"
COMMON_SHA = "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83"
OLD_ROW = b"eshkol_commit\tfe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9\n"
NEW_ROW = b"eshkol_commit\t6803dbe24fb65eaad2901767bf010787517946ef\n"
LOCK_PATH = Path("/bundle/public-g2-6803.lock")
ADMISSION_PATH = Path("/bundle/public-g2-source-admission.json")
COMMON_TRANSFORMS = (
    (b'LOCK_FILE="${PROJECT_ROOT}/toolchain/eshkol.lock"',
     b'LOCK_FILE="/bundle/public-g2-6803.lock"'),
    (b'CMAKE_C_COMPILER:FILEPATH', b'CMAKE_C_COMPILER:STRING'),
    (b'CMAKE_CXX_COMPILER:FILEPATH', b'CMAKE_CXX_COMPILER:STRING'),
)
F_PINS = {
    "build-off/eshkol-run": "379143cdb5385b3cc5f64473628adcca6f32e138649e6d0f84bb11d397f23aa9",
    "build-off/CMakeCache.txt": "3a6a665413001f7c0aa052e8fccc50fcde032cd2ab4ee331e3514dae01eaaff8",
    "build-off/libeshkol-runtime.a": "320c90a19279422e54f54d2f06a2dae02ddadefe321e77b2c294addb4dd8ff10",
    "build-off/libeshkol-static.a": "f12058653b0884ab7d6c72d8a92d1558e1721c77387e8975158d37b71ce53ec9",
    "logs/off-configure.argv.raw": "605172db54923f3e7d240f4605a7ae51cb1b3aa1111b6f99aaf1dd81f73a9ff3",
    "logs/off-configure.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "logs/off-build.argv.raw": "58ad3bf4fdff254afe6aa23c81377ee4e5a96287e777aef70838b9d030e2d21f",
    "logs/off-build.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "SHA256SUMS-RUN": "b655e250faae040689703f5699dc3df05413df07277614d1e5ab5ae0e8da4d5e",
}
ARTIFACTS = tuple(name for name in F_PINS if name.startswith("build-off/"))
GENERATED = {
    f"build-{mode}/generated/llvm-selected-2cc6322eadfd/{name}":
        f"/usr/lib/llvm-21/include/{name}"
    for mode in ("asan", "normal", "off", "tsan", "ubsan")
    for name in ("llvm", "llvm-c")
}
ELF_NEEDED = frozenset((
    "libpng16.so.16", "libjpeg.so.8", "libwebp.so.7", "libLLVM.so.21.1",
    "libopenblas.so.0", "libcrypto.so.3", "libz.so.1", "libstdc++.so.6",
    "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2",
))
ELF_RUNPATH = "$ORIGIN/../lib/eshkol/runtime-deps:/usr/lib/llvm-21/lib:"
OFFLINE_ROOT_REVIEW_SHA = "f15c0f9789c21ccd151a3687a047e49691a7d39866b67ff17d095c8a0b1c6760"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordinary(path: Path) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"not a unique ordinary file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_uid, opened.st_gid,
                 opened.st_mode, opened.st_nlink, opened.st_size) ==
                (before.st_dev, before.st_ino, before.st_uid, before.st_gid,
                 before.st_mode, before.st_nlink, before.st_size),
                f"file identity changed: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read()
        after = path.lstat()
        require((after.st_dev, after.st_ino, after.st_uid, after.st_gid,
                 after.st_mode, after.st_nlink, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns) ==
                (before.st_dev, before.st_ino, before.st_uid, before.st_gid,
                 before.st_mode, before.st_nlink, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns),
                f"file changed during read: {path}")
        return raw
    finally:
        os.close(fd)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                          check=True, capture_output=True, timeout=30).stdout.decode().strip()


def compiler_source(source: Path) -> dict[str, str]:
    require(source == Path("/workspace") and not source.is_symlink(),
            "compiler source must be the original /workspace mount")
    head, tree = git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    require((head, tree) == (COMPILER_HEAD, COMPILER_TREE),
            "compiler commit/tree mismatch")
    require(git(source, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "compiler source is dirty")
    return {"head": head, "tree": tree}


def offline_review(path: Path) -> str:
    require(path.is_absolute() and path.name == "SHA256SUMS-ROOT-REVIEW" and
            digest(ordinary(path)) == OFFLINE_ROOT_REVIEW_SHA,
            "accepted F native-controls-only offline review missing or drifted")
    return OFFLINE_ROOT_REVIEW_SHA


def candidate_lock(path: Path, tracked: bytes,
                   expected_path: Path = LOCK_PATH) -> dict[str, object]:
    require(path == expected_path and path.is_absolute(),
            "wrong candidate lock path")
    require(digest(tracked) == TRACKED_LOCK_SHA and tracked.count(OLD_ROW) == 1 and
            tracked.count(b"eshkol_commit\t") == 1, "production lock drift")
    expected = tracked.replace(OLD_ROW, NEW_ROW)
    require(len(expected) == 477 and digest(expected) == CANDIDATE_LOCK_SHA,
            "external lock derivation drift")
    parent_before = path.parent.lstat()
    require(stat.S_ISDIR(parent_before.st_mode) and
            not stat.S_ISLNK(parent_before.st_mode),
            "candidate lock parent is not ordinary")
    raw = ordinary(path)
    parent_after = path.parent.lstat()
    require((parent_before.st_dev, parent_before.st_ino,
             parent_before.st_uid, parent_before.st_mode) ==
            (parent_after.st_dev, parent_after.st_ino,
             parent_after.st_uid, parent_after.st_mode),
            "candidate lock parent changed")
    require(raw == expected, "external lock differs from sole fe9-to-6803 row")
    st = path.stat()
    return {"sha256": digest(raw), "dev": st.st_dev, "ino": st.st_ino,
            "size": st.st_size}


def source_admission(path: Path, expected_sha: str,
                     forbidden_roots: tuple[Path, ...],
                     expected_path: Path = ADMISSION_PATH,
                     *, outer_bundle_root: Path | None = None) -> dict:
    """An outer reviewed launcher must pin expected_sha, not a free CLI caller."""
    require(path == expected_path and path.is_absolute() and
            re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None and
            all(path != root and root not in path.parents and
                path not in root.parents for root in forbidden_roots),
            "source admission path/hash is outside the reviewed boundary")
    parent_before = path.parent.lstat()
    require(stat.S_ISDIR(parent_before.st_mode) and
            not stat.S_ISLNK(parent_before.st_mode),
            "source admission parent is not ordinary")
    raw = ordinary(path)
    parent_after = path.parent.lstat()
    require((parent_before.st_dev, parent_before.st_ino,
             parent_before.st_uid, parent_before.st_mode) ==
            (parent_after.st_dev, parent_after.st_ino,
             parent_after.st_uid, parent_after.st_mode),
            "source admission parent changed")
    require(digest(raw) == expected_sha, "source admission digest drift")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source admission JSON malformed") from exc
    require(type(data) is dict and raw == canonical_json(data) and
            set(data) == {"format", "head", "tree", "tracked_manifest_sha256",
                          "tracked_file_count", "independent_review",
                          "root_review"} and
            data["format"] == "g3c4-public-g2-source-admission-v1" and
            all(type(data[key]) is str and
                re.fullmatch(f"[0-9a-f]{{{40 if key in ('head', 'tree') else 64}}}",
                             data[key]) is not None
                for key in ("head", "tree", "tracked_manifest_sha256")) and
            type(data["tracked_file_count"]) is int and
            data["tracked_file_count"] > 0,
            "source admission schema/identity malformed")
    logical_reviews = {
        "independent_review": Path(
            "/bundle/reviews/independent/SHA256SUMS-INDEPENDENT-SOURCE-REVIEW"),
        "root_review": Path("/bundle/reviews/root/SHA256SUMS-ROOT-REVIEW"),
    }
    if outer_bundle_root is not None:
        require(outer_bundle_root.is_absolute() and
                stat.S_ISDIR(outer_bundle_root.lstat().st_mode) and
                outer_bundle_root.resolve(strict=True) == outer_bundle_root and
                path == outer_bundle_root / "public-g2-source-admission.json",
                "outer review bundle root is not exact")
    for key in ("independent_review", "root_review"):
        review = data[key]
        require(type(review) is dict and set(review) == {"path", "sha256"} and
                type(review["path"]) is str and
                type(review["sha256"]) is str and
                re.fullmatch(r"[0-9a-f]{64}", review["sha256"]) is not None,
                "source admission review reference malformed")
        review_path = Path(review["path"])
        if outer_bundle_root is not None:
            require(review_path == logical_reviews[key],
                    "outer logical review path differs")
            review_path = outer_bundle_root / review_path.relative_to("/bundle")
        require(review_path.is_absolute() and
                review_path.name.startswith("SHA256SUMS-") and
                digest(ordinary(review_path)) == review["sha256"],
                "source admission review reference drift")
    return {**data, "admission_sha256": expected_sha}


def staged_common(original: bytes) -> bytes:
    require(digest(original) == COMMON_SHA, "production common drift")
    staged = original
    for old, new in COMMON_TRANSFORMS:
        require(staged.count(old) == 1 and new not in staged,
                "common transform is not unique")
        staged = staged.replace(old, new)
    return staged


def cache_fields(raw: bytes) -> dict[str, str]:
    lines = raw.decode("utf-8").splitlines()
    required = {
        "CMAKE_HOME_DIRECTORY:INTERNAL": "/workspace",
        "CMAKE_C_COMPILER:STRING": "/usr/bin/clang-21",
        "CMAKE_CXX_COMPILER:STRING": "/usr/bin/clang++-21",
        "CMAKE_BUILD_TYPE:STRING": "RelWithDebInfo",
        "ESHKOL_BUILD_TESTS:BOOL": "ON",
        "ESHKOL_BUILD_EXAMPLES:BOOL": "OFF",
        "ESHKOL_BUILD_AGENT_FFI:BOOL": "OFF",
        "ESHKOL_XLA_ENABLED:BOOL": "OFF",
        "ESHKOL_QUANTUM_ENABLED:BOOL": "OFF",
        "ESHKOL_TENSORCORE_ENABLED:BOOL": "OFF",
    }
    for key, value in required.items():
        found = [line.split("=", 1)[1] for line in lines
                 if line.startswith(key + "=")]
        require(found == [value], f"F cache {key} mismatch")
    return required


def nul_argv(raw: bytes) -> list[str]:
    require(raw.endswith(b"\0") and raw.count(b"\0") > 2,
            "incomplete NUL argv")
    words = raw[:-1].split(b"\0")
    require(all(words), "empty argv word")
    return [word.decode("utf-8") for word in words]


def receipts(output: Path) -> dict[str, object]:
    configure = nul_argv(ordinary(output / "logs/off-configure.argv.raw"))
    build = nul_argv(ordinary(output / "logs/off-build.argv.raw"))
    require(ordinary(output / "logs/off-configure.exit") == b"0\n" and
            ordinary(output / "logs/off-build.exit") == b"0\n",
            "F configure/build did not genuinely exit zero")
    require(configure.count("/workspace") == 1 and
            configure.count("/out/build-off") == 1 and
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in configure and
            "-DESHKOL_BUILD_TESTS=ON" in configure,
            "F configure argv changed")
    require(build.count("/out/build-off") == 1 and
            "eshkol-run" in build and "--target" in build,
            "F build argv changed")
    return {"configure_argv": configure, "build_argv": build}


def original_output(output: Path) -> dict[str, str]:
    require(output.is_absolute() and not output.is_symlink(),
            "F output must be an absolute ordinary directory")
    for rel, expected in F_PINS.items():
        require(digest(ordinary(output / rel)) == expected,
                f"F sealed member drift: {rel}")
    manifest = ordinary(output / "SHA256SUMS-RUN").decode().splitlines()
    listed: set[str] = set()
    for line in manifest:
        require(re.fullmatch(r"[0-9a-f]{64}  \./[^\n]+", line) is not None,
                "F manifest malformed")
        expected, relative = line.split("  ./", 1)
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and
                relative not in listed, "F manifest traversal or duplicate")
        listed.add(relative)
        require(digest(ordinary(output / relative)) == expected,
                f"F inventory member drift: {relative}")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if stat.S_ISREG(p.lstat().st_mode)}
    require(actual == listed | {"SHA256SUMS-RUN"},
            "F original ordinary-file membership drift")
    links = {}
    for relative, target in GENERATED.items():
        path = output / relative
        require(path.is_symlink() and os.readlink(path) == target,
                f"F generated symlink drift: {relative}")
        links[relative] = target
    actual_links = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if stat.S_ISLNK(p.lstat().st_mode)}
    require(actual_links == set(GENERATED), "extra F symlink")
    directory_rows = ordinary(output / "directory-inventory.tsv").decode().splitlines()
    require(directory_rows and directory_rows[0] == "path_hex",
            "F directory inventory header changed")
    directories = set()
    for encoded in directory_rows[1:]:
        require(re.fullmatch(r"(?:[0-9a-f]{2})+", encoded) is not None,
                "F directory inventory path malformed")
        relative = bytes.fromhex(encoded).decode("utf-8")
        require(relative not in directories and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts,
                "F directory inventory duplicate/traversal")
        directories.add(relative)
    actual_directories = {p.relative_to(output).as_posix() for p in output.rglob("*")
                          if stat.S_ISDIR(p.lstat().st_mode)}
    require(directories == actual_directories, "F directory membership drift")
    all_nodes = {p.relative_to(output).as_posix() for p in output.rglob("*")}
    require(all_nodes == actual | actual_links | actual_directories,
            "F original contains an unclassified filesystem node")
    cache_fields(ordinary(output / "build-off/CMakeCache.txt"))
    receipts(output)
    return {"inventory_sha256": F_PINS["SHA256SUMS-RUN"],
            "links": links}


def elf_members(readelf_raw: bytes) -> tuple[set[str], str]:
    raw = readelf_raw.decode("utf-8")
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", raw)
    runpath = re.findall(r"\(RUNPATH\).*?\[(.*?)\]", raw)
    require(len(needed) == len(set(needed)) and set(needed) == ELF_NEEDED and
            runpath == [ELF_RUNPATH], "F ELF dependency inventory drift")
    return set(needed), runpath[0]


def resolved_image_libs(readelf_raw: bytes, loader_argv: bytes,
                        loader_exit: bytes, loader_stdout: bytes,
                        loader_stderr: bytes, expected_tsv: bytes) -> list[dict]:
    needed, _ = elf_members(readelf_raw)
    require(nul_argv(loader_argv) == ["/lib64/ld-linux-x86-64.so.2", "--list",
                                         "/out/build-off/eshkol-run"] and
            loader_exit == b"0\n" and loader_stderr == b"" and
            len(loader_stdout) <= 65536, "loader receipt invalid")
    observed = {}
    for line in loader_stdout.decode().splitlines():
        match = re.match(r"\s*(\S+) => (/(?:[^ ]+)) \(0x[0-9a-fA-F]+\)", line)
        if match:
            name, path = match.groups()
        else:
            match = re.match(r"\s*(/\S+) \(0x[0-9a-fA-F]+\)", line)
            if not match:
                require(re.fullmatch(r"\s*linux-vdso\.so\.1 \(0x[0-9a-fA-F]+\)",
                                     line) is not None, "unknown loader line")
                continue
            path = match.group(1)
            name = Path(path).name
        require(name not in observed, "duplicate loader dependency")
        observed[name] = path
    require(needed <= set(observed), "direct ELF dependency missing from loader")
    expected = {}
    for line in expected_tsv.decode().splitlines():
        fields = line.split("\t")
        require(len(fields) == 3 and fields[0] not in expected and
                re.fullmatch(r"[0-9a-f]{64}", fields[2]) is not None,
                "image pin row malformed")
        expected[fields[0]] = (fields[1], fields[2])
    require(set(expected) == set(observed),
            "image pins differ from complete loader closure")
    result = []
    for name in sorted(observed):
        path, sha = expected[name]
        require(observed[name] == path and path.startswith("/") and
                digest(ordinary(Path(path).resolve())) == sha,
                f"image library identity mismatch: {name}")
        result.append({"name": name, "path": path, "sha256": sha})
    return result


def copy_f_artifacts(original: Path, destination: Path) -> dict[str, str]:
    require(destination.is_absolute() and not destination.exists(),
            "F copy destination must be fresh")
    destination.mkdir(mode=0o700)
    result = {}
    for relative in ARTIFACTS:
        src = original / relative
        raw = ordinary(src)
        dst = destination / Path(relative).name
        require(not dst.exists(), "duplicate copied F artifact")
        exclusive_bytes(dst, raw, stat.S_IMODE(src.stat().st_mode))
        require(digest(ordinary(dst)) == F_PINS[relative],
                "copied F artifact differs from original")
        result[dst.name] = F_PINS[relative]
    actual = {p.name for p in destination.iterdir()}
    require(actual == {Path(p).name for p in ARTIFACTS},
            "copied F artifact subset is not exact")
    return result


def copied_artifacts(destination: Path,
                     expected_path: Path = Path("/out/build-off")) -> dict[str, str]:
    require(destination == expected_path and destination.is_dir() and
            not destination.is_symlink(), "copied F build path mismatch")
    actual = {p.name for p in destination.iterdir()}
    require(actual == {Path(p).name for p in ARTIFACTS},
            "copied F artifact subset is not exact")
    result = {}
    for relative in ARTIFACTS:
        name = Path(relative).name
        require(digest(ordinary(destination / name)) == F_PINS[relative],
                f"copied F artifact drift: {name}")
        result[name] = F_PINS[relative]
    return result


def final_copied_artifacts(destination: Path, compatibility_sha: str,
                           expected_path: Path = Path("/out/build-off")) -> dict[str, str]:
    require(destination == expected_path and destination.is_dir() and
            not destination.is_symlink(), "final copied F build path mismatch")
    expected = {Path(path).name for path in ARTIFACTS}
    require({path.name for path in destination.iterdir()} ==
            expected | {"eshkol-transformer-provenance.tsv"},
            "final copied F artifact membership drift")
    result = {}
    for relative in ARTIFACTS:
        name = Path(relative).name
        require(digest(ordinary(destination / name)) == F_PINS[relative],
                f"final copied F artifact drift: {name}")
        result[name] = F_PINS[relative]
    require(digest(ordinary(destination / "eshkol-transformer-provenance.tsv")) ==
            compatibility_sha, "compatibility view drift")
    return result


def project_copy(source: Path, candidate: Path,
                 decision: dict,
                 expected_candidate: Path = Path("/candidate")) -> dict[str, str]:
    require(candidate == expected_candidate and candidate.is_dir() and
            not candidate.is_symlink() and not (candidate / ".git").exists(),
            "candidate source path or metadata mismatch")
    expected_head, expected_tree = decision["head"], decision["tree"]
    observed = git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    require(observed == [expected_head, expected_tree] and
            git(source, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "source checkout identity/cleanliness mismatch")
    listing = subprocess.run(["git", "--no-optional-locks", "-C", str(source),
                              "ls-files", "-s", "-z"], check=True,
                             capture_output=True, timeout=30).stdout
    names = set()
    common_sha = None
    manifest_rows = []
    for entry in listing.rstrip(b"\0").split(b"\0"):
        metadata, raw_name = entry.split(b"\t", 1)
        mode, _, stage = metadata.decode().split()
        require(mode in ("100644", "100755") and stage == "0",
                "source requires ordinary stage-zero tracked files")
        name = raw_name.decode("utf-8")
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and
                name not in names, "invalid tracked source path")
        names.add(name)
        original = ordinary(source / relative)
        manifest_rows.append(f"{name}\t{mode}\t{digest(original)}\n")
        if name == "scripts/common.sh":
            original = staged_common(original)
            common_sha = digest(original)
        require(ordinary(candidate / relative) == original and
                stat.S_IMODE((candidate / relative).stat().st_mode) ==
                (0o755 if mode == "100755" else 0o644),
                f"candidate tracked source drift: {name}")
    members = list(candidate.rglob("*"))
    actual = {p.relative_to(candidate).as_posix() for p in members
              if stat.S_ISREG(p.lstat().st_mode)}
    require(actual == names and common_sha is not None,
            "candidate source membership drift")
    expected_directories = {parent.as_posix() for name in names
                            for parent in list(Path(name).parents)[:-1]}
    actual_directories = {p.relative_to(candidate).as_posix() for p in members
                          if stat.S_ISDIR(p.lstat().st_mode)}
    require(actual_directories == expected_directories,
            "candidate source directory membership drift")
    require({p.relative_to(candidate).as_posix() for p in members} ==
            actual | actual_directories,
            "candidate source contains an unclassified filesystem node")
    manifest_sha = digest("".join(manifest_rows).encode())
    require(len(names) == decision["tracked_file_count"] and
            manifest_sha == decision["tracked_manifest_sha256"],
            "reviewed source content closure drift")
    require(git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines() ==
            observed and git(source, "status", "--porcelain=v1",
                             "--untracked-files=all") == "",
            "source checkout drift during verification")
    return {"head": expected_head, "tree": expected_tree,
            "admission_sha256": decision["admission_sha256"],
            "tracked_manifest_sha256": manifest_sha,
            "staged_common_sha256": common_sha,
            "tracked_file_count": str(len(names))}


def stage_project_copy(source: Path, candidate: Path,
                       decision: dict,
                       expected_candidate: Path = Path("/candidate")) -> dict[str, str]:
    require(candidate == expected_candidate and candidate.is_absolute() and
            not candidate.exists(), "project copy destination must be fresh")
    require(git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines() ==
            [decision["head"], decision["tree"]] and
            git(source, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "project source is not the selected clean tree")
    listing = subprocess.run(["git", "--no-optional-locks", "-C", str(source),
                              "ls-files", "-s", "-z"], check=True,
                             capture_output=True, timeout=30).stdout
    candidate.mkdir(mode=0o700)
    for entry in listing.rstrip(b"\0").split(b"\0"):
        metadata, raw_name = entry.split(b"\t", 1)
        mode, _, stage = metadata.decode().split()
        require(mode in ("100644", "100755") and stage == "0",
                "project copy requires ordinary tracked files")
        relative = Path(raw_name.decode("utf-8"))
        require(not relative.is_absolute() and ".." not in relative.parts,
                "project copy contains invalid path")
        raw = ordinary(source / relative)
        if relative.as_posix() == "scripts/common.sh":
            raw = staged_common(raw)
        destination = candidate / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        exclusive_bytes(destination, raw, 0o755 if mode == "100755" else 0o644)
    return project_copy(source, candidate, decision,
                        expected_candidate)


def version_output(*argv: str) -> str:
    process = subprocess.run(argv, capture_output=True, timeout=10)
    require(process.returncode == 0 and len(process.stdout) <= 16384 and
            len(process.stderr) <= 16384, f"tool version failed: {argv[0]}")
    return (process.stdout + process.stderr).decode("utf-8")


def tool_versions() -> dict[str, str]:
    clang = lambda tool: re.search(r"version (\d+(?:\.\d+)+)",
                                   version_output(tool, "--version")).group(1)
    first = lambda tool, *args: re.search(r"(\d+(?:\.\d+)+)",
                                          version_output(tool, *args)).group(1)
    versions = {
        "llvm_version": version_output("/usr/bin/llvm-config-21", "--version").strip(),
        "cc_version": clang("/usr/bin/clang-21"),
        "cxx_version": clang("/usr/bin/clang++-21"),
        "cmake_version": first("/usr/bin/cmake", "--version"),
        "ninja_version": first("/usr/bin/ninja", "--version"),
        "make_version": first("/usr/bin/make", "--version"),
        "bash_version": first("/usr/bin/bash", "--version"),
        "git_version": first("/usr/bin/git", "--version"),
    }
    require(versions["llvm_version"] == "21.1.8" and
            versions["cc_version"] == "21.1.8" and
            versions["cxx_version"] == "21.1.8",
            "LLVM/Clang tool versions differ from F")
    return versions


def compiler_binary_and_elf(binary: Path, recorded_readelf: bytes) -> dict[str, str]:
    require(digest(ordinary(binary)) == F_PINS["build-off/eshkol-run"],
            "copied compiler changed before execution")
    require(version_output(str(binary), "--version").strip() ==
            "Eshkol Compiler v1.3.4-evolve", "copied compiler version mismatch")
    process = subprocess.run(["/usr/bin/readelf", "-d", str(binary)],
                             capture_output=True, timeout=10)
    require(process.returncode == 0 and process.stderr == b"" and
            len(process.stdout) <= 65536 and
            process.stdout == recorded_readelf,
            "readelf receipt differs from copied compiler")
    needed, runpath = elf_members(process.stdout)
    return {"readelf_sha256": digest(process.stdout),
            "needed_count": str(len(needed)), "runpath": runpath}


def retain_cleanup_error(primary: BaseException, error: OSError,
                         location: str) -> None:
    """Keep the primary error on guest Python 3.10, with bounded cleanup facts."""
    fact = f"attestation {location} close failed: {error!s}"[:512]
    try:
        facts = getattr(primary, "_attestation_cleanup", ())
        primary._attestation_cleanup = (*facts, fact)
        sys.stderr.write(fact + "\n")
    except BaseException:
        pass


def exclusive_bytes(path: Path, raw: bytes, mode: int = 0o400) -> None:
    require(not path.exists(), f"attestation output already exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 mode)
    primary: BaseException | None = None
    try:
        offset = 0
        while offset < len(raw):
            try:
                written = os.write(fd, raw[offset:])
            except InterruptedError:
                continue
            require(written > 0, "attestation write made no progress")
            offset += written
        os.fchmod(fd, mode)
        os.fsync(fd)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            os.close(fd)
        except OSError as close_error:
            if primary is None:
                raise
            retain_cleanup_error(primary, close_error, "descriptor")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    primary = None
    try:
        os.fsync(directory)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            os.close(directory)
        except OSError as close_error:
            if primary is None:
                raise
            retain_cleanup_error(primary, close_error, "directory")


def compatibility_tsv(source: Path, binary_sha: str,
                      tool_versions: dict[str, str]) -> bytes:
    """A new test-only compatibility view, never an F/bootstrap receipt."""
    keys = ("llvm_version", "cc_version", "cxx_version", "cmake_version",
            "ninja_version", "make_version", "bash_version", "git_version")
    require(set(tool_versions) == set(keys), "tool version inventory incomplete")
    rows = {
        "format_version": "1", "eshkol_repository":
            "https://github.com/Gabriel-Kahen/eshkol.git",
        "eshkol_commit": COMPILER_HEAD, "eshkol_source_dir": str(source),
        "eshkol_binary_sha256": binary_sha,
        "cc_path": "/usr/bin/clang-21", "cxx_path": "/usr/bin/clang++-21",
        **tool_versions,
        "feature_profile": "run-f-off-RelWithDebInfo-testsON-test-only",
        "attestation_origin": "new-test-only-F-copy;not-bootstrap;original-F-failed",
    }
    require(len(rows) == 17, "compatibility schema changed")
    return "".join(f"{key}\t{value}\n" for key, value in rows.items()).encode()


def prepared_payload(compiler: dict, project: dict, original: dict,
                     copied: dict, lock: dict, offline: str) -> dict:
    return {"format": "g3c4-F-copy-prepared-v1",
            "origin": "new-test-only;historical-F-remains-failed",
            "compiler": compiler, "project": project,
            "original": original, "copied": copied, "lock": lock,
            "offline_review_sha256": offline}


def canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def verify_prepared(raw: bytes, expected: dict) -> None:
    require(raw == canonical_json(expected),
            "prepared attestation missing, forged or changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "attest"))
    parser.add_argument("--f-output", type=Path, required=True)
    parser.add_argument("--compiler-source", type=Path, required=True)
    parser.add_argument("--copy-build", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--offline-review", type=Path, required=True)
    parser.add_argument("--project-source", type=Path, required=True)
    parser.add_argument("--outer-stage-child", action="store_true")
    parser.add_argument("--project-checkout", type=Path, required=True)
    parser.add_argument("--project-admission", type=Path, required=True)
    parser.add_argument("--project-admission-sha256", required=True)
    parser.add_argument("--readelf-raw", type=Path)
    parser.add_argument("--loader-argv", type=Path)
    parser.add_argument("--loader-exit", type=Path)
    parser.add_argument("--loader-stdout", type=Path)
    parser.add_argument("--loader-stderr", type=Path)
    parser.add_argument("--image-pins", type=Path)
    parser.add_argument("--image-pins-sha256")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    candidate_path = (Path("/stage/candidate") if args.outer_stage_child
                      else Path("/candidate"))
    require(os.environ.get("ESHKOL_ALLOW_UNSUPPORTED_HOST", "0") == "0",
            "unsupported-host bypass forbidden")
    require(args.copy_build == Path("/out/build-off") and
            args.project_source == candidate_path and
            args.evidence.is_absolute(),
            "F and project copy paths must match reviewed cache/profile")
    compiler = compiler_source(args.compiler_source)
    offline = offline_review(args.offline_review)
    original = original_output(args.f_output)
    tracked_lock = ordinary(args.project_checkout / "toolchain/eshkol.lock")
    lock = candidate_lock(args.candidate_lock, tracked_lock)
    forbidden = (args.project_source, args.project_checkout, args.f_output,
                 args.copy_build, args.compiler_source)
    decision = source_admission(args.project_admission,
                                args.project_admission_sha256, forbidden,
                                outer_bundle_root=(Path("/bundle")
                                                   if args.outer_stage_child else None))
    if args.phase == "prepare":
        require(not args.evidence.exists() and
                all(getattr(args, name) is None for name in
                    ("readelf_raw", "loader_argv", "loader_exit",
                     "loader_stdout", "loader_stderr", "image_pins",
                     "image_pins_sha256")),
                "prepare requires fresh evidence and no loader observations")
        args.evidence.mkdir(mode=0o700)
        project = stage_project_copy(args.project_checkout, args.project_source,
                                     decision, candidate_path)
        copied = copy_f_artifacts(args.f_output, args.copy_build)
        require(original_output(args.f_output) == original and
                compiler_source(args.compiler_source) == compiler and
                project_copy(args.project_checkout, args.project_source,
                             decision, candidate_path) == project and
                copied_artifacts(args.copy_build) == copied and
                candidate_lock(args.candidate_lock, tracked_lock) == lock and
                source_admission(args.project_admission,
                                 args.project_admission_sha256, forbidden,
                                 outer_bundle_root=(Path("/bundle")
                                                    if args.outer_stage_child else None)) == decision and
                offline_review(args.offline_review) == offline,
                "prepare inputs drifted")
        prepared = prepared_payload(compiler, project, original, copied,
                                    lock, offline)
        raw = canonical_json(prepared)
        exclusive_bytes(args.evidence / "F-copy-prepared.json", raw)
        print(raw.decode(), end="")
        return
    require(args.evidence.is_dir() and all(getattr(args, name) is not None for name in
            ("readelf_raw", "loader_argv", "loader_exit", "loader_stdout",
             "loader_stderr", "image_pins", "image_pins_sha256")),
            "attest requires prepared evidence and complete image receipts")
    project = project_copy(args.project_checkout, args.project_source,
                           decision, candidate_path)
    copied = copied_artifacts(args.copy_build)
    verify_prepared(ordinary(args.evidence / "F-copy-prepared.json"),
                    prepared_payload(compiler, project, original, copied,
                                     lock, offline))
    require(digest(ordinary(args.image_pins)) == args.image_pins_sha256,
            "image dependency pin file differs from external admission")
    readelf_raw = ordinary(args.readelf_raw)
    elf = compiler_binary_and_elf(args.copy_build / "eshkol-run", readelf_raw)
    image = resolved_image_libs(
        readelf_raw, ordinary(args.loader_argv),
        ordinary(args.loader_exit), ordinary(args.loader_stdout),
        ordinary(args.loader_stderr), ordinary(args.image_pins))
    versions = tool_versions()
    require(args.evidence != args.copy_build and
            not (args.copy_build / "eshkol-transformer-provenance.tsv").exists() and
            not (args.evidence / "F-copy-attestation.json").exists(),
            "attestation outputs must be fresh")
    compat = compatibility_tsv(args.compiler_source, F_PINS["build-off/eshkol-run"],
                               versions)
    receipt = {"format": "g3c4-F-copy-test-attestation-v1",
               "origin": "new-test-only;historical-F-remains-failed",
               "compiler": compiler, "project": project,
               "original": original, "copied": copied, "lock": lock,
               "offline_review_sha256": offline,
               "prepared_sha256": digest(ordinary(args.evidence / "F-copy-prepared.json")),
               "image_libraries": image,
               "image_pins_sha256": args.image_pins_sha256,
               "loader_stdout_sha256": digest(ordinary(args.loader_stdout)),
               "elf": elf,
               "tool_versions": versions,
               "compatibility_tsv_sha256": digest(compat)}
    raw = canonical_json(receipt)
    # Recheck selected identities before publishing either new file.
    require(original_output(args.f_output) == original and
            copied_artifacts(args.copy_build) == copied and
            compiler_source(args.compiler_source) == compiler and
            project_copy(args.project_checkout, args.project_source,
                         decision, candidate_path) == project and
            candidate_lock(args.candidate_lock, tracked_lock) == lock and
            source_admission(args.project_admission,
                             args.project_admission_sha256, forbidden,
                             outer_bundle_root=(Path("/bundle")
                                                if args.outer_stage_child else None)) == decision and
            offline_review(args.offline_review) == offline and
            digest(ordinary(args.image_pins)) == args.image_pins_sha256 and
            compiler_binary_and_elf(args.copy_build / "eshkol-run",
                                    ordinary(args.readelf_raw)) == elf and
            resolved_image_libs(
                ordinary(args.readelf_raw), ordinary(args.loader_argv),
                ordinary(args.loader_exit), ordinary(args.loader_stdout),
                ordinary(args.loader_stderr), ordinary(args.image_pins)) == image and
            tool_versions() == versions,
            "attestation inputs drifted before publication")
    exclusive_bytes(args.copy_build / "eshkol-transformer-provenance.tsv", compat)
    exclusive_bytes(args.evidence / "F-copy-attestation.json", raw)
    require(original_output(args.f_output) == original and
            final_copied_artifacts(args.copy_build, digest(compat)) == copied and
            compiler_source(args.compiler_source) == compiler and
            project_copy(args.project_checkout, args.project_source,
                         decision, candidate_path) == project and
            candidate_lock(args.candidate_lock, tracked_lock) == lock and
            source_admission(args.project_admission,
                             args.project_admission_sha256, forbidden,
                             outer_bundle_root=(Path("/bundle")
                                                if args.outer_stage_child else None)) == decision and
            offline_review(args.offline_review) == offline and
            digest(ordinary(args.image_pins)) == args.image_pins_sha256 and
            compiler_binary_and_elf(args.copy_build / "eshkol-run",
                                    ordinary(args.readelf_raw)) == elf and
            resolved_image_libs(
                ordinary(args.readelf_raw), ordinary(args.loader_argv),
                ordinary(args.loader_exit), ordinary(args.loader_stdout),
                ordinary(args.loader_stderr), ordinary(args.image_pins)) == image and
            tool_versions() == versions and
            ordinary(args.evidence / "F-copy-attestation.json") == raw,
            "attestation inputs or published receipt drifted")
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
