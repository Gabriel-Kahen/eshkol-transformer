"""Inactive E3 extraction adapter for the reviewed F-loader J lifecycle.

There is deliberately no CLI or decision pin here. A later ROOT-reviewed
wrapper must authenticate the exact J source, supply a one-use decision, and
own both locks for the complete Docker and physical-closure interval.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from tests.e3_private import inactive_extraction_recipe as recipe


J_HEAD = "07a2ae53cbdf5620fd10319b8c95bf0261dfeb51"
J_TREE = "c5a3b90a2e4c9be388141531f1fb31579c4ea324"
J_SOURCE = "70428fe98636c609a59e7c1385179dc88629f08e66376229004fc24171a60644"
J_ROOT = "da1fd8d09b59002a6ab2c45a7781635207d979f51635163939fa6c754a3a67c9"
J_IND = "c15b54878d3e70744b9a8e34add72dc0e45d329f13d12ef7f2b44462c8bb6a86"
J_FILES = {
    "tests/g3c4/f_loader_outer.py": "73066210494ef0cf3f5bd824520fc70e73545f4176a6ae3eedabbe16d5a97360",
    "tests/g3c4/f_artifact_attest.py": "5aac55665ad5a1767c54591ec3530667cced310f07948257521f34ed43999e78",
    "tests/g3c4/f_image_loader_collect.py": "8b92aabe7fb742b81dc5e1d8fcd57bf92b9f5b3ad702920e9c826ce8faafd101",
}
RECIPE_SHA = "c36001659fafebcaf48ecff4dccdc46c9ea2db4f639721d7dd59aa0f47ab36c6"
MATERIAL = recipe.MATERIAL_NAMES
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"e3-sealed-fixture-extraction-[a-z0-9-]+\Z")
GUEST = ["-B", "/bundle/e3_extraction_guest.py", "--material-pins-sha256"]


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def exact_bytes(path: Path, expected: str, ordinary) -> bytes:
    if not HEX64.fullmatch(expected):
        raise ValueError("missing immutable pin")
    raw = ordinary(path)
    if digest(raw) != expected:
        raise ValueError(f"selected bytes changed: {path}")
    return raw


def h_source(source: Path, ordinary) -> None:
    """Authenticate reviewed J code and independent/root seals before import."""
    if not source.is_absolute() or not stat.S_ISDIR(source.lstat().st_mode) or \
            source.resolve(strict=True) != source:
        raise ValueError("J source directory is not canonical")
    for name, expected in J_FILES.items():
        exact_bytes(source / name, expected, ordinary)
    exact_bytes(source / "SHA256SUMS-SOURCE", J_SOURCE, ordinary)
    exact_bytes(source / "SHA256SUMS-ROOT-REVIEW", J_ROOT, ordinary)
    exact_bytes(source / "SHA256SUMS-INDEPENDENT-SOURCE-REVIEW", J_IND, ordinary)


def material(bundle: Path, expected: str, ordinary) -> tuple[dict[str, bytes], dict[str, str]]:
    raw = exact_bytes(bundle / "e3-material-pins.json", expected, ordinary)
    pins = json.loads(raw)
    if type(pins) is not dict or set(pins) != MATERIAL or any(
            type(value) is not str or not HEX64.fullmatch(value)
            for value in pins.values()) or raw != (
            json.dumps(pins, sort_keys=True, separators=(",", ":")).encode() + b"\n"):
        raise ValueError("complete canonical E3 material pins required")
    bodies = {name: exact_bytes(bundle / "e3-material" / name, pins[name], ordinary)
              for name in sorted(MATERIAL)}
    return bodies, pins


def flat_attester(bundle: Path, ordinary) -> None:
    """Pin the guest's separate flat F utility before Docker creation."""
    exact_bytes(bundle / "f_artifact_attest.py", recipe.F_ATTEST_SHA, ordinary)


def selected(h, decision_path: Path, decision_sha: str, *, fresh: bool) -> dict[str, Any]:
    """E3-specific admission; retain H's host/source and socket predicates."""
    if not HEX64.fullmatch(decision_sha):
        raise ValueError("external ROOT decision pin required")
    h.require(decision_path.is_absolute() and
              decision_path.resolve(strict=True) == decision_path,
              "ROOT decision path is not canonical")
    raw = exact_bytes(decision_path, decision_sha, h.ordinary)
    decision = json.loads(raw)
    expected = {"format", "name", "image", "docker_sha256", "socket_dev", "socket_ino",
                "compiler", "source", "f_output", "bundle", "candidate", "out",
                "tool_pins_sha256", "runtime_pins_sha256", "offline_review_sha256",
                "e3_material_pins_sha256", "h_source_sha256", "guest_sha256",
                "recipe_sha256"}
    h.require(type(decision) is dict and set(decision) == expected and
              decision["format"] == "e3-sealed-fixture-extraction-outer-v1" and
              decision["image"] == h.IMAGE and type(decision["name"]) is str and
              NAME.fullmatch(decision["name"]) and
              raw == h.canonical(decision), "E3 decision schema or canonical bytes")
    for key in ("docker_sha256", "tool_pins_sha256", "runtime_pins_sha256",
                "offline_review_sha256", "e3_material_pins_sha256", "guest_sha256",
                "recipe_sha256"):
        h.require(type(decision[key]) is str and HEX64.fullmatch(decision[key]),
                  f"decision pin malformed: {key}")
    h.require(decision["h_source_sha256"] == J_SOURCE and
              decision["recipe_sha256"] == RECIPE_SHA and
              decision["tool_pins_sha256"] == h.collector.TOOL_PINS_SHA and
              decision["runtime_pins_sha256"] == h.collector.RUNTIME33_SHA and
              decision["offline_review_sha256"] == h.attest.OFFLINE_ROOT_REVIEW_SHA,
              "selected H/tool/runtime provenance")
    h.require(all(type(decision[key]) is int and decision[key] > 0
                  for key in ("socket_dev", "socket_ino")),
              "rootless socket decision malformed")
    for key in ("compiler", "source", "f_output", "bundle", "candidate", "out"):
        h.require(type(decision[key]) is str, "decision path malformed")
        h.directory(Path(decision[key]))
    roots = [Path(decision[key]).resolve() for key in
             ("compiler", "source", "f_output", "bundle", "candidate", "out")]
    h.require(len(set(roots)) == 6 and all(
        not (a in b.parents or b in a.parents) for i, a in enumerate(roots)
        for b in roots[i + 1:]), "selected roots overlap")
    h.require(digest(h.ordinary(h.DOCKER)) == decision["docker_sha256"],
              "Docker client drift")
    socket = Path(f"/run/user/{h.os.getuid()}/docker.sock").lstat()
    h.require(h.stat.S_ISSOCK(socket.st_mode) and socket.st_uid == h.os.getuid() and
              (socket.st_dev, socket.st_ino) == (decision["socket_dev"],
                                               decision["socket_ino"]),
              "rootless socket drift")
    head, tree = h.attest.git(Path(decision["compiler"]), "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    h.require((head, tree) == (h.attest.COMPILER_HEAD, h.attest.COMPILER_TREE) and
              h.attest.git(Path(decision["compiler"]), "status", "--porcelain=v1",
                           "--untracked-files=all") == "", "compiler source drift")
    bundle = Path(decision["bundle"])
    h_source(bundle / "h-source", h.ordinary)
    h.require(digest(h.ordinary(bundle / "accepted-tools.tsv")) == h.collector.TOOL_PINS_SHA and
              digest(h.ordinary(bundle / "accepted-runtime33.tsv")) == h.collector.RUNTIME33_SHA and
              digest(h.ordinary(bundle / "offline/SHA256SUMS-ROOT-REVIEW")) ==
              h.attest.OFFLINE_ROOT_REVIEW_SHA, "tool/runtime bundle drift")
    h.attest.candidate_lock(bundle / "public-g2-6803.lock",
                             h.ordinary(Path(decision["source"]) / "toolchain/eshkol.lock"),
                             bundle / "public-g2-6803.lock")
    flat_attester(bundle, h.ordinary)
    h.attest.original_output(Path(decision["f_output"]))
    bodies, pins = material(bundle, decision["e3_material_pins_sha256"], h.ordinary)
    admission = recipe.e3_admission(bodies["e3_admission"], pins["e3_admission"],
                                     bodies["e3_root_review"], bodies["e3_ind_review"])
    recipe.loader_proof(bodies["loader_attestation"], bodies["loader_compatibility"],
                        bodies["loader_readelf"], pins["loader_attestation"],
                        pins["loader_compatibility"], pins["loader_readelf"])
    h.source_state(Path(decision["source"]), admission)
    exact_bytes(bundle / "e3_extraction_guest.py", decision["guest_sha256"], h.ordinary)
    exact_bytes(bundle / "e3_recipe.py", RECIPE_SHA, h.ordinary)
    if fresh:
        for key in ("candidate", "out"):
            h.require(not any(Path(decision[key]).iterdir()), f"fresh {key} is not empty")
    return decision


def create_argv(h, original_create, decision: dict[str, Any]) -> list[str]:
    """Retain the exact H Docker profile and six mounts; replace only Cmd."""
    prior = dict(decision, source_admission_sha256="0" * 64)
    argv = original_create(prior)
    old = ["-B", "-m", "tests.g3c4.f_loader_guest",
           "--source-admission-sha256", "0" * 64]
    h.require(argv[-len(old):] == old, "H guest tail changed")
    return argv[:-len(old)] + GUEST + [decision["e3_material_pins_sha256"]]


def inspect_owned(h, raw: bytes, decision: dict[str, Any], cid: str,
                  running: bool | None, image: bytes) -> dict:
    """Authenticate authentic E3 Cmd, then all unchanged H profile fields."""
    data = json.loads(raw)
    h.require(type(data) is list and len(data) == 1, "inspect cardinality")
    row = data[0]
    config, host, state = row["Config"], row["HostConfig"], row["State"]
    image_row = json.loads(image)
    h.require(type(image_row) is list and len(image_row) == 1 and
              image_row[0]["Id"] == h.IMAGE and
              image_row[0]["Config"].get("Labels") ==
              {"org.opencontainers.image.version": "22.04"},
              "immutable f31 image labels changed")
    labels = dict(image_row[0]["Config"]["Labels"])
    h.require("eshkol.gate" not in labels, "image reserves selected gate label")
    labels["eshkol.gate"] = decision["name"]
    h.require(row["Id"] == cid and row["Name"] == "/" + decision["name"] and
              config["Labels"] == labels and
              row["Image"] == h.IMAGE and
              (running is None or state["Running"] is running) and
              config["User"] == "0:0" and config["WorkingDir"] == "/source" and
              config["Entrypoint"] == ["/usr/bin/python3.10"] and
              config["Cmd"] == GUEST + [decision["e3_material_pins_sha256"]] and
              h.env_map(config["Env"]) == h.expected_env(image) and
              host["ReadonlyRootfs"] is True and host["NetworkMode"] == "none" and
              host["Tmpfs"] == {"/tmp": "rw,nosuid,nodev,size=268435456,mode=1777"} and
              host["Memory"] == 4 * 1024**3 and host["MemorySwap"] == 4 * 1024**3 and
              host["PidsLimit"] == 128 and host["NanoCpus"] == 2_000_000_000 and
              host["Ulimits"] == [{"Name": "fsize", "Hard": h.FSIZE_LIMIT,
                                   "Soft": h.FSIZE_LIMIT}], "owned E3 inspect profile")
    roles = (("compiler", "/workspace", True), ("source", "/source", True),
             ("f_output", "/f-output", True), ("bundle", "/bundle", True),
             ("candidate", "/stage", False), ("out", "/out", False))
    expected = {decision[key]: (target, readonly) for key, target, readonly in roles}
    mounts = {entry["Source"]: (entry["Destination"], not entry["RW"])
              for entry in row["Mounts"] if entry["Type"] == "bind"}
    h.require(mounts == expected and len(row["Mounts"]) == 6 and
              all(entry["Mode"] == "" and entry["Propagation"] == "rprivate"
                  for entry in row["Mounts"]), "owned E3 mounts differ")
    host_mounts = [{"Type": "bind", "Source": decision[key], "Target": target,
                    **({"ReadOnly": True} if readonly else {})}
                   for key, target, readonly in roles]
    h.require(host["Mounts"] == host_mounts and host["Binds"] is None,
              "owned E3 host mounts differ")
    return row


def guest_output(h, decision: dict[str, Any]) -> None:
    out = Path(decision["out"])
    h.directory(out)
    h.directory(out / "build-off")
    h.directory(out / "phases")
    h.attest.copied_artifacts(out / "build-off", out / "build-off")
    for mode in ("normal", "sanitize"):
        phase = out / "phases" / mode
        h.directory(phase)
        emission = out / f"emissions-{mode}"
        h.directory(emission)
        status = json.loads(h.ordinary(phase / "status.json"))
        h.require(status["returncode"] == 0 and status["quiescent"] is True and
                  not status["failure"] and h.ordinary(phase / "stderr.raw") == b"" and
                  status["argv"] == list(recipe.guest_command(mode)),
                  f"E3 extraction {mode} receipt failed")
        extracted = emission / "e3_sealed_fixture_package.o.extract"
        h.directory(extracted)
        h.require(h.ordinary(extracted / "STATUS") == (
            b"EXTRACTION_ONLY: emitted bytes require separate review; no package admitted\n") and
            not os.path.lexists(emission / "e3_sealed_fixture_package.o") and
            not os.path.lexists(emission / "e3_sealed_fixture_package.o.evidence"),
            "E3 emission is not extraction-only")
    h.directory(out / "evidence")
    receipt = json.loads(h.ordinary(out / "evidence" / "extraction-guest.json"))
    h.require(receipt == {"format": "e3-sealed-fixture-extraction-guest-v1",
                          "modes": ["normal", "sanitize"],
                  "material_pins_sha256": decision["e3_material_pins_sha256"],
                  "status": "EXTRACTION_ONLY"}, "E3 guest evidence drift")


def bind_and_run(h, h_root: Path, decision_path: Path, decision_sha: str,
                 evidence: Path, monitor) -> None:
    """Inactive adapter entry; an external wrapper must own lease/heavy/once."""
    h_source(h_root, h.ordinary)
    decision = json.loads(exact_bytes(decision_path, decision_sha, h.ordinary))
    if h_root != Path(decision["bundle"]) / "h-source":
        raise ValueError("J source is not the selected read-only bundle")
    if Path(h.__file__).resolve(strict=True) != (
            h_root / "tests/g3c4/f_loader_outer.py").resolve(strict=True):
        raise ValueError("wrong J lifecycle module")
    replacements = {
        "selected": lambda path, pin, *, fresh: selected(h, path, pin, fresh=fresh),
        "create_argv": lambda decision: create_argv(h, original_create, decision),
        "inspect_owned": lambda raw, decision, cid, running, image: inspect_owned(
            h, raw, decision, cid, running, image),
        "guest_output": lambda decision: guest_output(h, decision),
    }
    old = {key: getattr(h, key) for key in replacements}
    original_create = old["create_argv"]
    try:
        for key, replacement in replacements.items():
            setattr(h, key, replacement)
        h.run_once(decision_path, decision_sha, evidence, monitor)
    finally:
        for key, original in old.items():
            setattr(h, key, original)


def contract() -> dict[str, object]:
    return {"status": "SOURCE_ONLY_INACTIVE", "h_head": J_HEAD, "h_tree": J_TREE,
            "h_source_sha256": J_SOURCE, "decision_format":
            "e3-sealed-fixture-extraction-outer-v1", "mounts": recipe.MOUNTS,
            "guest_sequence": [recipe.guest_command("normal"),
                               recipe.guest_command("sanitize")],
            "missing_runtime_authority": ["actual F loader result/reviews and E3 admission",
                                          "ROOT fixed wrapper, lease/heavy owner, once decision"]}
