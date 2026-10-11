#!/usr/bin/env python3
"""Inert, noninstalled E3 N1 raw-capture preflight and receipt validator.

No subprocess, container, extracted binary, or target is ever executed here.
The source-reviewed pin is deliberately absent until independently supplied.
"""

import argparse
import hashlib
import json
import os
import resource
import shutil
import stat
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

# F's image ID names an OCI index in the local content store.
F_IMAGE = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
F_SOURCE = "94b8fecd2d187ba990128c44884dc35a40d130e9"
F_TREE = "d28cffc1a9a8de3c78b2a1cbd9ea27d1cb5fdd1e"
F_COMPILER = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
F_FAILED_SEAL = "8269be7bf960758bfccf1d374df39c9d89824a6882007fac235f88ec02f86a83"
F_BUNDLE = Path("/home/gabe/e3-numerical-transcript-94b8fec-fe9-f31-20261002-prepared-f")
F_SMALL_FILES = {
    "SHA256SUMS-FAILED": F_FAILED_SEAL,
    "plan.tsv": "b74f0e1b699fe12fd71f74e3b96424051e068373996f3b076f8d418bf93964e4",
    "launch.sh": "8a2e39a8ff180d8cd8efd0bea0cc8536da9389ec054ba8ce62089558298ca346",
}
F_OBJECTS = {"private.o": 25, "package-native-5.o": 2, "package-native-6.o": 12}
F_OBJECT_SHA256 = {
    "private.o": "ad1fd0ae2363a44fcb35092f5aa625d9831ee24c4fc82d53a5572abe86c27007",
    "package-native-5.o": "6997483f01975defd030e394e43cc2edbff6dc852262c700d144fe1c0f573b35",
    "package-native-6.o": "bd85cace584cb0e1b8333c658c3508d1d83439ab710560805a375b9cd2c22458",
}
CAPTURE_CATEGORIES = {"call-relocations", "final-producer", "abi-declarations",
                      "provider-elf", "provider-symlinks", "loader-observation",
                      "system-headers", "argv-env"}
REVIEWED_PLAN_SHA256 = None  # A source change and independent review must set this.
MAX_JSON = 2 * 1024 * 1024
MAX_RECEIPT = 4 * 1024 * 1024
CHUNK = 1024 * 1024


class Hold(Exception):
    pass


def need(condition, code):
    if not condition:
        raise Hold(code)


def sha256_file(path, byte_cap, deadline):
    h = hashlib.sha256()
    count = 0
    with open(path, "rb") as stream:
        while data := stream.read(CHUNK):
            count += len(data)
            need(count <= byte_cap, "byte-cap")
            need(time.monotonic() <= deadline, "time-cap")
            h.update(data)
    return h.hexdigest(), count


def read_json(path, cap):
    need(path.is_file() and not path.is_symlink(), "missing-or-symlink-json")
    need(path.stat().st_size <= cap, "json-byte-cap")
    def no_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, "duplicate-json-key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)


def digest(value):
    need(isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
         and all(c in "0123456789abcdef" for c in value[7:]), "bad-digest")
    return value[7:]


def blob(layout, descriptor, byte_cap, deadline):
    d = descriptor["digest"]
    name = digest(d)
    path = layout / "blobs" / "sha256" / name
    need(path.is_file() and not path.is_symlink(), "missing-oci-blob:" + name)
    actual, size = sha256_file(path, byte_cap, deadline)
    need(actual == name and size == descriptor["size"], "oci-blob-drift:" + name)
    return path


def safe_guest_path(value):
    need(isinstance(value, str) and value.startswith("/") and "\x00" not in value,
         "bad-guest-path")
    parts = PurePosixPath(value).parts
    need(".." not in parts and "." not in parts and value == "/" + "/".join(parts[1:]),
         "bad-guest-path")
    return value.lstrip("/")


def rss_check(cap):
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= cap, "rss-cap")


def scan_selected_layers(paths, selected, byte_cap, deadline, rss_cap=2 * 1024**3):
    """Read tar entries only; reject whiteouts/hardlinks and ambiguous selected paths."""
    targets = {safe_guest_path(s["path"]): s for s in selected}
    need(len(targets) == len(selected), "duplicate-selected-path")
    seen = {}
    read_count = 0
    for layer_number, path in enumerate(paths):
        with open(path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|*") as archive:
            for member in archive:
                need(time.monotonic() <= deadline, "time-cap")
                rss_check(rss_cap)
                name = member.name
                while name.startswith("./"):
                    name = name[2:]
                if name.startswith("/") or ".." in PurePosixPath(name).parts:
                    raise Hold("unsafe-oci-entry")
                if name.startswith(".wh.") or "/.wh." in name:
                    base = name.rsplit("/", 1)
                    parent = base[0] if len(base) == 2 else ""
                    whiteout = (parent + "/" if parent else "") + base[-1][4:]
                    need(not any(t == whiteout or t.startswith(whiteout + "/")
                                 or base[-1] == ".wh..wh..opq" and
                                 (not parent or t.startswith(parent + "/")) for t in targets),
                         "selected-whiteout")
                if name not in targets:
                    continue
                need(member.isfile() or member.issym(), "selected-unsupported-entry")
                entry = {"layer": layer_number, "path": "/" + name,
                         "kind": "symlink" if member.issym() else "file",
                         "mode": member.mode & 0o7777}
                if member.issym():
                    entry["target"] = member.linkname
                else:
                    stream = archive.extractfile(member)
                    need(stream is not None, "selected-unreadable")
                    h = hashlib.sha256()
                    size = 0
                    while chunk := stream.read(CHUNK):
                        size += len(chunk)
                        read_count += len(chunk)
                        need(read_count <= byte_cap, "selected-byte-cap")
                        need(time.monotonic() <= deadline, "time-cap")
                        h.update(chunk)
                    entry.update({"sha256": h.hexdigest(), "size": size})
                seen[name] = entry
    need(set(seen) == set(targets), "missing-selected-guest-file")
    for name, entry in seen.items():
        need(entry == {k: v for k, v in targets[name].items() if k != "role"},
             "selected-guest-drift:" + name)
    for name, entry in seen.items():
        if entry["kind"] != "symlink":
            continue
        target = entry["target"]
        resolved = PurePosixPath(target.lstrip("/")) if target.startswith("/") else PurePosixPath(name).parent / target
        parts = []
        for part in resolved.parts:
            if part == "..":
                need(bool(parts), "symlink-escapes-image")
                parts.pop()
            elif part != ".":
                parts.append(part)
        need("/".join(parts) in seen, "symlink-target-unpinned")
    return len(seen)


def validate_routes(routes):
    need(isinstance(routes, list) and len(routes) == 7, "mount-route-count")
    fixed = [
        ("/home/gabe/.codex/venvs/eshkol-oracle-py3146", "/home/gabe/.codex/venvs/eshkol-oracle-py3146", "ro"),
        ("/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "/home/gabe/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu", "ro"),
        ("/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "ro"),
        ("/home/gabe/.codex/worktrees/e3-numerical-stable-source/eshkol-transformer", "/workspace", "ro"),
        ("/home/gabe/.codex/worktrees/wave3-runtime-repin/eshkol-transformer/.deps", "/workspace/.deps", "ro"),
        ("/home/gabe/Documents/ChatGPT/eshkol-transformer/.git", "/home/gabe/Documents/ChatGPT/eshkol-transformer/.git", "ro"),
    ]
    for route, expected in zip(routes[:6], fixed):
        need((route.get("host"), route.get("guest"), route.get("mode")) == expected,
             "mount-route-drift")
    out = routes[6]
    need(out.get("guest") == "/out" and out.get("mode") == "rw"
         and isinstance(out.get("host"), str) and os.path.isabs(out["host"]),
         "output-route-drift")
    need(len({x["guest"] for x in routes}) == 7, "duplicate-guest-route")


def validate_host_inputs(plan, deadline):
    limits = plan["limits"]
    roots = plan["mount_roots"]
    need(isinstance(roots, list) and len(roots) == 7, "mount-root-provenance-count")
    need([x["path"] for x in roots] == [x["host"] for x in plan["mounts"]],
         "mount-root-provenance-drift")
    for item in roots:
        path = Path(item["path"])
        mode = path.lstat().st_mode
        kind = "symlink" if stat.S_ISLNK(mode) else "directory" if stat.S_ISDIR(mode) else "file"
        need(kind == item["kind"] and stat.S_IMODE(mode) == item["mode"],
             "mount-root-drift")
        if kind == "symlink":
            need(os.readlink(path) == item["target"], "mount-root-link-drift")
        need(str(path.resolve(strict=True)) == item["resolved"], "mount-root-resolution-drift")
    selected = plan["selected_host_files"]
    need(isinstance(selected, list) and selected and
         {"tool", "interpreter", "loader"} <= {x["role"] for x in selected},
         "host-file-role-missing")
    for item in selected:
        path = Path(item["path"])
        need(path.is_absolute() and path.is_file() and not path.is_symlink(),
             "host-file-missing")
        need(stat.S_IMODE(path.stat().st_mode) == item["mode"], "host-file-mode-drift")
        actual, size = sha256_file(path, limits["max_raw_record_bytes"], deadline)
        need((actual, size) == (item["sha256"], item["size"]), "host-file-drift")


def validate_capture_recipe(plan):
    commands = plan["capture_commands"]
    need(isinstance(commands, list) and
         {x["category"] for x in commands} == CAPTURE_CATEGORIES,
         "capture-category-coverage")
    pinned_tools = {x["path"] for x in plan["selected_guest_files"] if x["role"] == "tool"}
    for command in commands:
        need(command["tool"] in pinned_tools and
             isinstance(command["argv"], list) and command["argv"] and
             command["argv"][0] == command["tool"] and
             all(isinstance(x, str) and "\x00" not in x for x in command["argv"]),
             "capture-argv-unpinned")
        need(isinstance(command["env"], dict) and
             all(isinstance(k, str) and isinstance(v, str) and "\x00" not in k + v
                 for k, v in command["env"].items()), "capture-env-unpinned")
        need(type(command["seconds"]) is int and
             0 < command["seconds"] <= plan["limits"]["seconds"] and
             type(command["max_output_bytes"]) is int and
             0 < command["max_output_bytes"] <= plan["limits"]["max_output_bytes"],
             "capture-command-cap")


def validate_oci(plan, deadline, expected_image=F_IMAGE):
    layout = Path(plan["oci_layout"])
    need(layout.is_absolute() and layout.is_dir() and not layout.is_symlink(), "missing-oci-layout")
    need(read_json(layout / "oci-layout", MAX_JSON) == {"imageLayoutVersion": "1.0.0"},
         "oci-layout-version")
    index = layout / "index.json"
    index_hash, _ = sha256_file(index, MAX_JSON, deadline)
    need(index_hash == plan["index_sha256"] and index_hash == digest(expected_image),
         "oci-index-drift")
    indices = read_json(index, MAX_JSON)["manifests"]
    manifest_desc = plan["manifest"]
    amd64 = [item for item in indices if item.get("platform") ==
             {"architecture": "amd64", "os": "linux"}]
    need(len(amd64) == 1 and manifest_desc == amd64[0], "oci-amd64-manifest-drift")
    max_blob = plan["limits"]["max_blob_bytes"]
    manifest_path = blob(layout, manifest_desc, MAX_JSON, deadline)
    manifest = read_json(manifest_path, MAX_JSON)
    need(manifest["config"] == plan["config"] and manifest["layers"] == plan["layers"],
         "oci-descriptor-drift")
    config_path = blob(layout, manifest["config"], MAX_JSON, deadline)
    config = read_json(config_path, MAX_JSON)
    need(config.get("architecture") == "amd64" and config.get("os") == "linux", "image-platform-drift")
    need(config.get("rootfs", {}).get("diff_ids") == plan["diff_ids"], "oci-diffid-drift")
    need(len(plan["layers"]) == len(plan["diff_ids"]), "oci-layer-count")
    need(all(layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar"
             and layer["digest"] == diff_id
             for layer, diff_id in zip(plan["layers"], plan["diff_ids"])),
         "compressed-or-unmatched-layer-unsupported")
    paths = [blob(layout, layer, max_blob, deadline) for layer in plan["layers"]]
    return scan_selected_layers(paths, plan["selected_guest_files"],
                                plan["limits"]["max_selected_bytes"], deadline,
                                plan["limits"]["max_rss_bytes"])


def validate_capture_receipt(receipt_path, plan, deadline):
    """Validate independently captured raw records; never infer their semantics."""
    receipt = read_json(receipt_path, MAX_RECEIPT)
    need(receipt.get("schema") == 1 and receipt.get("status") == "complete", "raw-receipt-incomplete")
    required = ("call_sites", "provider_graph", "loader_observation", "system_headers",
                "argv_env", "declarations", "f_object_parity")
    need(all(receipt.get(k) for k in required), "raw-receipt-missing-section")
    rows = receipt["call_sites"]
    need(isinstance(rows, list) and len(rows) == 39, "call-count")
    counts = {name: 0 for name in F_OBJECTS}
    identities = set()
    for row in rows:
        key = (row["object"], row["section"], row["offset"])
        need(key not in identities and row["object"] in counts, "call-identity")
        identities.add(key)
        counts[row["object"]] += 1
        need(all(row.get(k) for k in ("producer", "abi", "declaration", "raw_records")),
             "call-claim-incomplete")
    need(counts == F_OBJECTS, "call-object-count")
    required_parity = {x["object"]: x["sha256"] for x in plan["f_objects"]}
    need(receipt["f_object_parity"] == required_parity, "f-object-parity")
    records = receipt.get("raw_records")
    need(isinstance(records, list) and records, "raw-records-missing")
    referenced = {p for row in rows for p in row["raw_records"]}
    for key in required[1:-1]:
        section = receipt[key]
        need(isinstance(section, list) and all(isinstance(p, str) for p in section),
             "raw-section-shape:" + key)
        referenced.update(section)
    need(referenced <= {r["path"] for r in records}, "call-raw-path-missing")
    for record in records:
        path = Path(record["path"])
        need(path.is_absolute() and path.is_file() and not path.is_symlink(), "raw-path-missing")
        actual, size = sha256_file(path, plan["limits"]["max_raw_record_bytes"], deadline)
        need((actual, size) == (record["sha256"], record["size"]), "raw-record-drift")
    need(receipt.get("reviewed_semantics") is False, "self-attested-review")
    return len(records)


def preflight(plan_path, receipt_path=None, reviewed_pin=REVIEWED_PLAN_SHA256,
              output_host=None):
    plan_path = Path(plan_path)
    need(reviewed_pin is not None, "reviewed-plan-pin-absent")
    plan_hash, _ = sha256_file(plan_path, MAX_JSON, time.monotonic() + 5)
    need(plan_hash == reviewed_pin, "reviewed-plan-drift")
    plan = read_json(plan_path, MAX_JSON)
    need(plan.get("schema") == 1 and plan.get("image_id") == F_IMAGE
         and plan.get("source_head") == F_SOURCE and plan.get("source_tree") == F_TREE
         and plan.get("compiler_commit") == F_COMPILER
         and plan.get("failed_seal_sha256") == F_FAILED_SEAL, "f-identity-drift")
    limits = plan["limits"]
    need(all(type(limits.get(k)) is int and 0 < limits[k] <= ceiling for k, ceiling in {
        "seconds": 1800, "max_blob_bytes": 8 * 1024**3,
        "max_selected_bytes": 2 * 1024**3,
        "max_raw_record_bytes": 2 * 1024**3,
        "max_output_bytes": 2 * 1024**3,
        "max_rss_bytes": 2 * 1024**3,
    }.items()), "resource-limits")
    deadline = time.monotonic() + limits["seconds"]
    for name, expected in F_SMALL_FILES.items():
        actual, _ = sha256_file(F_BUNDLE / name, MAX_JSON, deadline)
        need(actual == expected, "f-seal-or-launch-drift:" + name)
    need(output_host is None or str(Path(output_host).resolve()) == plan["output_host"],
         "output-host-argument-drift")
    need(shutil.disk_usage(Path(plan["output_host"]).parent).free >=
         2 * limits["max_output_bytes"], "free-space-cap")
    validate_routes(plan["mounts"])
    need(plan["mounts"][-1]["host"] == plan["output_host"], "output-host-route-drift")
    validate_host_inputs(plan, deadline)
    validate_capture_recipe(plan)
    need(all(x.get("role") in ("tool", "interpreter", "loader", "provider", "header")
             for x in plan["selected_guest_files"]), "selected-role")
    need({"tool", "interpreter", "loader", "provider", "header"}
         <= {x["role"] for x in plan["selected_guest_files"]}, "selected-role-missing")
    count = validate_oci(plan, deadline)
    need(len(plan["f_objects"]) == 3 and
         {x["object"]: x["sha256"] for x in plan["f_objects"]} == F_OBJECT_SHA256,
         "f-object-manifest")
    for item in plan["f_objects"]:
        actual, _ = sha256_file(Path(item["path"]), limits["max_raw_record_bytes"], deadline)
        need(actual == item["sha256"], "f-object-drift")
    need(receipt_path is not None, "capture-receipt-absent")
    raw_count = validate_capture_receipt(Path(receipt_path), plan, deadline)
    return {"selected_guest_files": count, "raw_records": raw_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--capture", action="store_true", help="reserved; always HOLD")
    parser.add_argument("--out", required=True, help="fresh directory for bounded HOLD receipt")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    result = {"schema": 1, "status": "HOLD", "action": "none", "reason": None}
    try:
        need(not args.capture, "capture-disabled-by-source")
        result["counts"] = preflight(args.plan, args.receipt, output_host=args.out)
        result["reason"] = "semantic-review-and-capture-runner-not-enabled"
    except (Hold, KeyError, ValueError, TypeError, OSError, tarfile.TarError) as exc:
        result["reason"] = str(exc)[:256]
    target = out / "preflight.json"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(target, flags, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(target)
    return 1


if __name__ == "__main__":
    sys.exit(main())
