"""Measure the unchanged heavy-gate resource floors; never launch a build."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def measure():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if fields[0] in {"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"}:
            if len(fields) != 3 or fields[2] != "kB":
                raise ValueError("unexpected memory measurement units")
            memory[fields[0][:-1]] = int(fields[1])
    release = dict(
        line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines()
        if "=" in line
    )
    docker_root = subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True,
        timeout=30,
    ).strip()
    if not docker_root.startswith("/"):
        raise ValueError("Docker root is not an absolute path")
    workspace = Path.cwd().resolve()
    disk = {
        "workspace": {"path": str(workspace), "free_bytes": shutil.disk_usage(workspace).free},
        "docker_root": {"path": docker_root, "free_bytes": shutil.disk_usage(docker_root).free},
    }
    supported = (
        release.get("ID", "").strip('"') == "ubuntu"
        and release.get("VERSION_ID", "").strip('"') == "22.04"
        and platform.machine() == "x86_64"
    )
    floors = {"mem_available_kib": 16777216, "swap_free_kib": 8388608,
              "workspace_free_bytes": 34359738368, "docker_root_free_bytes": 34359738368}
    checks = {
        "supported_os_arch": supported,
        "mem_available": memory["MemAvailable"] >= floors["mem_available_kib"],
        "swap_free": memory["SwapFree"] >= floors["swap_free_kib"],
        "workspace_free": disk["workspace"]["free_bytes"] >= floors["workspace_free_bytes"],
        "docker_root_free": disk["docker_root"]["free_bytes"] >= floors["docker_root_free_bytes"],
    }
    return {
        "schema": "g3r-hosted-resource-probe-v1",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_image": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "architecture": platform.machine(),
        "os_release": release,
        "memory_kib": memory,
        "disk": disk,
        "floors": floors,
        "checks": checks,
        "resource_floor_met": all(checks.values()),
        "scope": "Resource measurement before toolchain installation; no build or runtime admission.",
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.mkdir(mode=0o700)
    data = (json.dumps(measure(), indent=2, sort_keys=True) + "\n").encode()
    (output / "resource-probe.json").write_bytes(data)
    (output / "SHA256SUMS").write_text(
        hashlib.sha256(data).hexdigest() + "  resource-probe.json\n"
    )
    print(data.decode(), end="")
