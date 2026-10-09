"""Inactive E3 sealed-fixture emission recipe; no launcher or admission API."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import stat


E3_COMMIT = "e7db77c6258801203f8970d600867db6d7730bd6"
E3_TREE = "fe3822da4a3cd0a673b881c008954ce52316a5d0"
E3_SOURCE_MANIFEST = "def101a5bfd696e3d7258c19c767faa9f8ab7b0f42709fed3d349cc07b636ee4"
E3_ROOT_REVIEW = "18872c984bdd38541a6e316d3527891815d5338f5ca47347f13c8b76e6ea2d07"
E3_IND_REVIEW = "52b7a2fa6e2920d2b06511575496d36328829baa276caf00f03774d4e1df872a"
E3_TRACKED_COUNT = 2391
E3_TRACKED_SHA = "8ee9edddd39d481e0a8cb62843a645ab86014e1bf074e9e683428a00c2e3b2f3"
F_ATTEST_SHA = "5aac55665ad5a1767c54591ec3530667cced310f07948257521f34ed43999e78"
COMMON_SHA = "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83"
EXTERNAL_LOCK_SHA = "b798822b18535405092b201248d8e37682d5ddb9555b6080ee32131659bfbea5"
TRANSFORMS = (
    (b'LOCK_FILE="${PROJECT_ROOT}/toolchain/eshkol.lock"',
     b'LOCK_FILE="/bundle/public-g2-6803.lock"'),
    (b'CMAKE_C_COMPILER:FILEPATH', b'CMAKE_C_COMPILER:STRING'),
    (b'CMAKE_CXX_COMPILER:FILEPATH', b'CMAKE_CXX_COMPILER:STRING'),
)
MOUNTS = (
    ("compiler-source", "/workspace", "ro"),
    ("historical-f-output", "/f-output", "ro"),
    ("admitted-e3-checkout", "/source", "ro"),
    ("staged-e3-parent", "/stage", "rw"),
    ("external-lock-and-reviews", "/bundle", "ro"),
    ("persistent-emissions", "/out", "rw"),
)
MISSING_RUNTIME_AUTHORITY = (
    "accepted copied-F loader result and exact F compatibility TSV",
    "full tracked E3 source admission and staged-common attestation",
    "current image/tool/runtime/source/lease/heavy admission and one-shot decision",
)


def common_candidate(original: bytes) -> bytes:
    if sha256(original).hexdigest() != COMMON_SHA:
        raise ValueError("reviewed common.sh bytes changed")
    staged = original
    for before, after in TRANSFORMS:
        if staged.count(before) != 1 or after in staged:
            raise ValueError("copied-F common transform is not unique")
        staged = staged.replace(before, after)
    return staged


def guest_command(mode: str) -> tuple[str, ...]:
    if mode not in ("normal", "sanitize"):
        raise ValueError("exact extraction modes only")
    return (
        "/usr/bin/env", "ESHKOL_SOURCE_DIR=/workspace",
        "ESHKOL_BUILD_DIR=/out/build-off",
        "ESHKOL_ALLOW_UNSUPPORTED_HOST=0",
        "E3_COMPILER_TIMEOUT_SECONDS=900", "/usr/bin/bash",
        "/stage/candidate/scripts/extract-e3-sealed-fixture-inventories.sh",
        f"/out/emissions-{mode}", mode,
    )


def e3_admission(raw: bytes, expected_sha: str,
                 root_review: bytes, independent_review: bytes) -> dict[str, object]:
    """Read-only, E3-specific admission; caller must supply a ROOT-fixed SHA."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or \
            sha256(raw).hexdigest() != expected_sha:
        raise ValueError("E3 source admission is not externally pinned")
    if sha256(root_review).hexdigest() != E3_ROOT_REVIEW or \
            sha256(independent_review).hexdigest() != E3_IND_REVIEW:
        raise ValueError("E3 source reviews changed")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("E3 source admission malformed") from error
    expected = {
        "format": "e3-sealed-fixture-source-admission-v1",
        "head": E3_COMMIT, "tree": E3_TREE,
        "tracked_file_count": E3_TRACKED_COUNT,
        "tracked_manifest_sha256": E3_TRACKED_SHA,
        "source_manifest_sha256": E3_SOURCE_MANIFEST,
        "root_source_review_sha256": E3_ROOT_REVIEW,
        "independent_source_review_sha256": E3_IND_REVIEW,
    }
    if type(data) is not dict or data != expected or \
            raw != json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n":
        raise ValueError("E3 source admission schema or canonical bytes changed")
    return {**data, "admission_sha256": expected_sha}


def loader_proof(attestation: bytes, compatibility: bytes, readelf: bytes,
                 expected_attestation_sha: str, expected_compatibility_sha: str,
                 expected_readelf_sha: str) -> dict[str, object]:
    """Authenticate selected F-copy fields; full loader closure is external."""
    for raw, expected in ((attestation, expected_attestation_sha),
                          (compatibility, expected_compatibility_sha),
                          (readelf, expected_readelf_sha)):
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(raw).hexdigest() != expected:
            raise ValueError("actual loader material lacks external fixed pin")
    try:
        data = json.loads(attestation)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F loader attestation malformed") from error
    if type(data) is not dict or type(data.get("compiler")) is not dict or \
            attestation != json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n" or \
            data.get("format") != "g3c4-F-copy-test-attestation-v1" or \
            data.get("origin") != "new-test-only;historical-F-remains-failed" or \
            data.get("compatibility_tsv_sha256") != expected_compatibility_sha or \
            data.get("compiler", {}).get("head") != "6803dbe24fb65eaad2901767bf010787517946ef" or \
            data.get("compiler", {}).get("tree") != "539d22b1b6a89c290aa13c91263982332b01b87b":
        raise ValueError("F loader result is not the reviewed copied-compiler contract")
    if type(data.get("tool_versions")) is not dict or type(data.get("elf")) is not dict:
        raise ValueError("F loader tool or ELF proof is absent")
    lines = compatibility.decode("utf-8").splitlines()
    rows = [line.split("\t", 1) for line in lines]
    versions = ("llvm_version", "cc_version", "cxx_version", "cmake_version",
                "ninja_version", "make_version", "bash_version", "git_version")
    expected_keys = {"format_version", "eshkol_repository", "eshkol_commit",
                     "eshkol_source_dir", "eshkol_binary_sha256", "cc_path",
                     "cxx_path", "feature_profile", "attestation_origin", *versions}
    rowmap = dict(rows) if all(len(row) == 2 for row in rows) else {}
    if len(rows) != 17 or any(len(row) != 2 for row in rows) or \
            set(rowmap) != expected_keys or \
            rowmap.get("format_version") != "1" or \
            rowmap.get("eshkol_repository") != "https://github.com/Gabriel-Kahen/eshkol.git" or \
            rowmap.get("eshkol_commit") != "6803dbe24fb65eaad2901767bf010787517946ef" or \
            rowmap.get("eshkol_source_dir") != "/workspace" or \
            rowmap.get("eshkol_binary_sha256") != "379143cdb5385b3cc5f64473628adcca6f32e1386496d0f84bb11d397f23aa9" or \
            rowmap.get("cc_path") != "/usr/bin/clang-21" or \
            rowmap.get("cxx_path") != "/usr/bin/clang++-21" or \
            rowmap.get("feature_profile") != "run-f-off-RelWithDebInfo-testsON-test-only" or \
            rowmap.get("attestation_origin") != "new-test-only-F-copy;not-bootstrap;original-F-failed" or \
            {key: rowmap.get(key) for key in versions} != data["tool_versions"]:
        raise ValueError("F compatibility TSV is not the selected 17-row view")
    return data


def accepted_attester(path: Path):
    """Load only the byte-identical reviewed F utility, never a tool fallback."""
    if path != Path("/bundle/f_artifact_attest.py"):
        raise ValueError("reviewed F attester absent or changed")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("reviewed F attester is not ordinary")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        raw = os.read(fd, 1024 * 1024 + 1)
        after = path.lstat()
    finally:
        os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                              value.st_nlink, value.st_uid, value.st_size)
    if identity(before) != identity(opened) or identity(before) != identity(after) or \
            len(raw) > 1024 * 1024 or sha256(raw).hexdigest() != F_ATTEST_SHA:
        raise ValueError("reviewed F attester bytes or identity changed")
    spec = importlib.util.spec_from_file_location("accepted_f_attester", path)
    if spec is None or spec.loader is None:
        raise ValueError("reviewed F attester cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_source_and_compiler(attester, admission: dict[str, object],
                                loader: dict[str, object],
                                compatibility: bytes, readelf: bytes) -> dict[str, object]:
    """Inactive callable; a later ROOT-pinned wrapper owns any invocation."""
    checkout = Path("/source")
    candidate = Path("/stage/candidate")
    f_output = Path("/f-output")
    compiler_source = Path("/workspace")
    build = Path("/out/build-off")
    if candidate.exists() or build.exists():
        raise ValueError("candidate and copied compiler must be fresh")
    attester.compiler_source(compiler_source)
    attester.original_output(f_output)
    attester.offline_review(Path("/bundle/offline/SHA256SUMS-ROOT-REVIEW"))
    attester.candidate_lock(Path("/bundle/public-g2-6803.lock"),
                            attester.ordinary(checkout / "toolchain/eshkol.lock"))
    if attester.git(checkout, "rev-parse", "HEAD", "HEAD^{tree}").splitlines() != \
            [E3_COMMIT, E3_TREE] or \
            attester.git(checkout, "status", "--porcelain=v1",
                         "--untracked-files=all") != "":
        raise ValueError("E3 checkout changed before staging")
    staged = attester.stage_project_copy(checkout, candidate, admission, candidate)
    copied = attester.copy_f_artifacts(f_output, build)
    attester.exclusive_bytes(build / "eshkol-transformer-provenance.tsv",
                             compatibility)
    if attester.final_copied_artifacts(build, sha256(compatibility).hexdigest(), build) != copied or \
            attester.compiler_binary_and_elf(build / "eshkol-run", readelf) != loader["elf"] or \
            attester.project_copy(checkout, candidate, admission, candidate) != staged:
        raise ValueError("E3 copied compiler or staged source drift")
    return {"status": "PREPARED_FOR_EXTRACTION_ONLY", "staged_source": staged,
            "copied_compiler": copied, "compatibility_sha256": sha256(compatibility).hexdigest()}


MATERIAL_NAMES = frozenset((
    "e3_admission", "e3_root_review", "e3_ind_review", "loader_manifest",
    "loader_attestation", "loader_compatibility", "loader_readelf",
    "loader_root_review", "loader_ind_review",
))


def prepare(material: dict[str, bytes], fixed_pins: dict[str, str]) -> dict[str, object]:
    """No CLI: only a later separately reviewed ROOT wrapper may fix these pins."""
    if set(material) != MATERIAL_NAMES or set(fixed_pins) != MATERIAL_NAMES:
        raise ValueError("complete actual E3 and loader authority is missing")
    for name in MATERIAL_NAMES:
        raw, expected = material[name], fixed_pins[name]
        if type(raw) is not bytes or not raw or \
                type(expected) is not str or \
                not re.fullmatch(r"[0-9a-f]{64}", expected) or \
                sha256(raw).hexdigest() != expected:
            raise ValueError(f"external reviewed pin missing or drifted: {name}")
    admission = e3_admission(material["e3_admission"], fixed_pins["e3_admission"],
                             material["e3_root_review"], material["e3_ind_review"])
    loader = loader_proof(
        material["loader_attestation"], material["loader_compatibility"],
        material["loader_readelf"], fixed_pins["loader_attestation"],
        fixed_pins["loader_compatibility"], fixed_pins["loader_readelf"])
    attester = accepted_attester(Path("/bundle/f_artifact_attest.py"))
    return prepare_source_and_compiler(attester, admission, loader,
                                       material["loader_compatibility"],
                                       material["loader_readelf"])


def contract() -> dict[str, object]:
    return {
        "status": "SOURCE_ONLY_INACTIVE",
        "e3_commit": E3_COMMIT,
        "e3_tree": E3_TREE,
        "source_manifest_sha256": E3_SOURCE_MANIFEST,
        "f_attester_sha256": F_ATTEST_SHA,
        "external_lock_sha256": EXTERNAL_LOCK_SHA,
        "mounts": MOUNTS,
        "sequence": (guest_command("normal"), guest_command("sanitize")),
        "missing_runtime_authority": MISSING_RUNTIME_AUTHORITY,
    }
