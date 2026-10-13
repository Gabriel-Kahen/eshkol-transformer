#!/usr/bin/env python3
"""One-use v2 tracer calibration, carried by the pinned C7e lifecycle launcher."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
VESSEL = Path("/home/gabe/.codex/evidence/eshkol-transformer/"
              "g3-p2g2-c7e-executable-successor-20261012-d/launcher.py")
VESSEL_SHA256 = "ee9fb1e75b4cb3cdf21a96bc1092cf26e6a4bfb9a3a7087408afcf92dc4a7f5a"
HEAD = "2f867bed5ec4f2667bc5cd4cd68f7b47dbea37af"
TREE = "f4ff635c25caef7452a1bbc01cdc58cb3385e16f"
IMAGE = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
FILES = frozenset({"README.md", "plan.json", "stage-matrix.tsv", "launcher.py",
                   "bootstrap.py", "RECOVERY-LOCK", "entry.sh", "postflight.py",
                   "test_offline.py", "SHA256SUMS-PREPARED"})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vessel():
    if not VESSEL.is_file() or VESSEL.is_symlink() or sha(VESSEL) != VESSEL_SHA256:
        raise ValueError("pinned lifecycle launcher differs")
    spec = importlib.util.spec_from_file_location("g3c4_c7e_lifecycle", VESSEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FILES = FILES
    module.EXPECTED_HEAD = HEAD
    module.EXPECTED_TREE = TREE
    module.verify_inputs = verify_inputs
    original_seal = module.seal_result

    def seal_calibration(bundle, result, outcome):
        if outcome.get("status") == "COMPILED_GATE_EXECUTED_UNADMITTED":
            outcome = {"status": "V2_TRACE_CALIBRATION_EXECUTED_UNADMITTED",
                       "guest_exit": outcome["guest_exit"],
                       "limitations": ["terminal/Eshkol gate unrun",
                                       "failed exec attempts and child streams unproven"]}
        original_seal(bundle, result, outcome)

    module.seal_result = seal_calibration
    return module


def verify_inputs(bundle, plan, *, mountpoint=False):
    core = vessel()
    if plan.get("kind") != "g3c4-p2g2-c7i-v2-trace-calibration" or \
            plan.get("source_head") != HEAD or plan.get("source_tree") != TREE or \
            plan.get("image_id") != IMAGE or \
            plan.get("image_tag") != "eshkol-checked-promotion-llvm21:20260922" or \
            plan.get("gate_command") != ["bash", "/instrument/entry.sh"] or \
            plan.get("sole_slot_lock") != str(core.SOLE_SLOT) or \
            plan.get("guest_source_mount") != "/workspace:ro" or \
            plan.get("guest_deps_mount") != "/workspace/.deps:ro" or \
            type(plan.get("launch_enabled")) is not bool:
        raise ValueError("calibration authority or image pin changed")
    source = Path(plan["source_worktree"])
    deps = Path(plan["deps_root"])
    if source.is_symlink() or not source.is_dir() or deps != source / ".deps" or \
            deps.is_symlink() or not deps.is_dir() or any(deps.iterdir()):
        raise ValueError("exact source/dependency mount differs")
    core.regular(source / ".git")
    if (source / ".git").read_text() != plan["source_gitfile"] or \
            core.git(source, "rev-parse", "HEAD") != HEAD or \
            core.git(source, "rev-parse", "HEAD^{tree}") != TREE or \
            core.git(source, "rev-parse", "--git-common-dir") != plan["source_git_common"] or \
            core.git(source, "branch", "--show-current") or \
            core.git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("detached C7i source identity differs")
    core.full_tracked_tree(source)
    expected = plan["source_files"]
    if set(expected) != {"tools/g3c4_native_trace/trace.c",
                         "tools/g3c4_native_trace/sha256.c",
                         "tools/g3c4_native_trace/sha256.h",
                         "tools/g3c4_native_trace/tests/canary.c",
                         "scripts/g3c4_native_trace_receipt.py"}:
        raise ValueError("calibration source inventory differs")
    for name, digest in expected.items():
        path = source / name
        core.regular(path)
        if sha(path) != digest:
            raise ValueError(f"calibration source changed: {name}")
    if plan.get("lifecycle_path") != str(VESSEL) or \
            plan.get("lifecycle_sha256") != VESSEL_SHA256 or \
            plan.get("bootstrap_sha256") != sha(bundle / "bootstrap.py"):
        raise ValueError("lifecycle/bootstrap pin changed")
    if plan.get("docker_cli_path") != "/home/gabe/bin/docker" or \
            sha(Path(plan["docker_cli_path"])) != plan.get("docker_cli_sha256") or \
            plan.get("git_path") != "/usr/bin/git" or \
            sha(Path(plan["git_path"])) != plan.get("git_sha256") or \
            plan.get("python_path") != os.path.realpath(sys.executable) or \
            sha(Path(plan["python_path"])) != plan.get("python_sha256"):
        raise ValueError("host toolchain pin changed")
    socket = Path(plan["docker_socket"])
    info = socket.lstat()
    if socket.is_symlink() or not stat.S_ISSOCK(info.st_mode) or \
            (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != \
            (plan["docker_socket_uid"], plan["docker_socket_gid"],
             plan["docker_socket_mode"]):
        raise ValueError("Docker endpoint identity differs")
    if not 60 <= plan.get("wall_seconds", 0) <= 900 or \
            plan.get("cli_memory_bytes") != 536870912 or \
            plan.get("container_memory_bytes") != 8589934592 or \
            plan.get("container_cpus") != 2 or \
            plan.get("min_free_disk_bytes") != 68719476736 or \
            plan.get("min_mem_available_kib") != 10485760:
        raise ValueError("calibration resource contract changed")


def prepare_enabled(target):
    core = vessel()
    core.pristine(BUNDLE)
    plan = core.load_plan(BUNDLE)
    verify_inputs(BUNDLE, plan)
    if plan["launch_enabled"]:
        raise ValueError("source template is already enabled")
    target = Path(target)
    if not target.is_absolute() or target.parent.is_symlink() or \
            target.exists() or target.is_symlink():
        raise ValueError("one-use destination already exists")
    target.mkdir(mode=0o700)
    try:
        for name in FILES - {"plan.json", "SHA256SUMS-PREPARED"}:
            shutil.copyfile(BUNDLE / name, target / name)
        plan["launch_enabled"] = True
        (target / "plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
        manifest = "".join(f"{sha(target / name)}  {name}\n"
                           for name in sorted(FILES - {"SHA256SUMS-PREPARED"}))
        (target / "SHA256SUMS-PREPARED").write_text(manifest)
        for name in FILES:
            with (target / name).open("rb") as stream:
                os.fsync(stream.fileno())
        core.fsync_dir(target)
        core.pristine(target)
        verify_inputs(target, plan)
    except BaseException:
        # A failed preparation is retained for inspection, never reused.
        raise
    print(f"prepared one-use calibration bundle: {target}")


def main():
    core = vessel()
    if sys.argv[1:] == ["--check-source"]:
        core.pristine(BUNDLE)
        verify_inputs(BUNDLE, core.load_plan(BUNDLE))
        print(f"C7i v2 calibration source PASS: {HEAD}; image and runtime UNRUN")
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--prepare-enabled":
        prepare_enabled(sys.argv[2])
        return 0
    if sys.argv[1:] == ["--launch"]:
        return core.run_live(BUNDLE)
    if sys.argv[1:] == ["--recover"]:
        return core.recover_failure(BUNDLE)
    raise ValueError("usage: launcher.py --check-source|--prepare-enabled DIR|--launch|--recover")


if __name__ == "__main__":
    raise SystemExit(main())
