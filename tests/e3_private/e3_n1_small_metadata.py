#!/usr/bin/env python3
"""Inert f32 OCI small-metadata acquisition candidate; no layer reads."""

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import stat
import sys
import time

F32_INDEX = "sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d"
F32_NAME = "eshkol-f32-quantum-llvm21:20260925"
REVIEWED_REQUEST_SHA256 = None  # A later source review must pin exact request bytes.
INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
TAR_TYPE = "application/vnd.oci.image.layer.v1.tar"
GZIP_TYPE = TAR_TYPE + "+gzip"
JSON_CAP = 2 * 1024 * 1024
EVIDENCE_CAP = 8 * 1024 * 1024
REQUEST_CAP = 1024 * 1024
CHUNK = 64 * 1024


class Hold(Exception):
    pass


def need(condition, reason):
    if not condition:
        raise Hold(reason)


def canonical(value):
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (UnicodeEncodeError, ValueError, TypeError, RecursionError) as error:
        raise Hold("noncanonical-value") from error


def parse(raw, require_canonical=False):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate-json-key")
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        need(math.isfinite(value), "nonfinite-json")
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda _: need(False, "nonfinite-json"),
                           parse_float=finite)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise Hold("invalid-json") from error
    # Also rejects escaped unpaired surrogates anywhere in OCI JSON.
    canonical(value)
    if require_canonical:
        need(raw == canonical(value), "noncanonical-request")
    return value


def fields(value, names, reason):
    need(type(value) is dict and set(value) == set(names), reason)


def digest(value):
    need(type(value) is str and len(value) == 71 and value.startswith("sha256:")
         and all(ch in "0123456789abcdef" for ch in value[7:]), "digest")


def positive(value, cap, reason):
    need(type(value) is int and 0 < value <= cap, reason)


def absolute(path):
    need(type(path) is str and path.startswith("/") and path != "/" and
         "\x00" not in path and all(part not in ("", ".", "..")
                                  for part in path.split("/")[1:]), "absolute-path")


def identity(info):
    return {"dev": info.st_dev, "ino": info.st_ino, "mode": info.st_mode,
            "uid": info.st_uid, "gid": info.st_gid, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def current_rss_bytes():
    """Current Linux resident pages, independent of inherited ru_maxrss peaks."""
    fd = os.open("/proc/self/statm", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw = os.read(fd, 129)
    finally:
        os.close(fd)
    parts = raw.split()
    need(len(raw) <= 128 and len(parts) == 7 and
         all(part.isdigit() for part in parts), "rss-measurement-unavailable")
    page_size = os.sysconf("SC_PAGE_SIZE")
    need(type(page_size) is int and page_size > 0, "rss-page-size")
    return int(parts[1]) * page_size


def open_nofollow(path):
    """Return a regular file fd and its full directory identity chain."""
    absolute(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    parts = path.split("/")[1:]
    parent = os.open("/", flags | os.O_DIRECTORY)
    chain = [{"path": "/", **identity(os.fstat(parent))}]
    try:
        current = ""
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
            current += "/" + part
            chain.append({"path": current, **identity(os.fstat(parent))})
        fd = os.open(parts[-1], flags, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise Hold("source-not-regular")
        return fd, chain
    finally:
        os.close(parent)


def read_pinned(path, cap, expected_identity=None, deadline=None, rss_cap=None):
    before = identity(os.lstat(path))
    fd, chain = open_nofollow(path)
    try:
        opened = identity(os.fstat(fd))
        need(before == opened and opened["size"] <= cap, "source-open-or-size-drift")
        if expected_identity is not None:
            fields(expected_identity, ("file", "ancestors"), "source-identity-fields")
            need(expected_identity == {"file": opened, "ancestors": chain},
                 "source-identity-drift")
        chunks = []
        count = 0
        while block := os.read(fd, CHUNK):
            count += len(block)
            need(count <= cap, "read-byte-cap")
            if deadline is not None:
                need(time.monotonic() <= deadline, "read-time-cap")
            if rss_cap is not None:
                need(current_rss_bytes() <= rss_cap,
                     "read-rss-cap")
            chunks.append(block)
        need(count == opened["size"] and identity(os.fstat(fd)) == opened,
             "source-fd-drift")
    finally:
        os.close(fd)
    after_fd, after_chain = open_nofollow(path)
    try:
        need(identity(os.fstat(after_fd)) == opened and after_chain == chain and
             identity(os.lstat(path)) == opened, "source-path-drift")
    finally:
        os.close(after_fd)
    return b"".join(chunks), {"file": opened, "ancestors": chain}


def descriptor(value, allowed):
    need(type(value) is dict and set(value) >= {"mediaType", "digest", "size"},
         "descriptor-fields")
    need(type(value["mediaType"]) is str and value["mediaType"] in allowed,
         "descriptor-media")
    digest(value["digest"])
    positive(value["size"], JSON_CAP if allowed == {MANIFEST_TYPE} or
             allowed == {CONFIG_TYPE} else 512 * 1024**2, "descriptor-size")


def selected_manifest(index):
    need(type(index) is dict and type(index.get("schemaVersion")) is int and
         index["schemaVersion"] == 2 and index.get("mediaType") == INDEX_TYPE and
         type(index.get("manifests")) is list, "index-schema")
    selected = [item for item in index["manifests"] if type(item) is dict and
                item.get("platform") == {"os": "linux", "architecture": "amd64"}]
    need(len(selected) == 1, "platform-selection")
    descriptor(selected[0], {MANIFEST_TYPE})
    return selected[0]


def manifest_config(manifest):
    need(type(manifest) is dict and type(manifest.get("schemaVersion")) is int and
         manifest["schemaVersion"] == 2 and
         manifest.get("mediaType") == MANIFEST_TYPE and
         type(manifest.get("layers")) is list and
         0 < len(manifest["layers"]) <= 32, "manifest-schema")
    descriptor(manifest.get("config"), {CONFIG_TYPE})
    for layer in manifest["layers"]:
        descriptor(layer, {TAR_TYPE, GZIP_TYPE})
    return manifest["config"], manifest["layers"]


def config_claims(config, count):
    need(type(config) is dict and config.get("os") == "linux" and
         config.get("architecture") == "amd64", "config-platform")
    rootfs = config.get("rootfs")
    need(type(rootfs) is dict and rootfs.get("type") == "layers" and
         type(rootfs.get("diff_ids")) is list and
         len(rootfs["diff_ids"]) == count, "config-rootfs")
    for item in rootfs["diff_ids"]:
        digest(item)
    return rootfs["diff_ids"]


def prior_bytes(prior, request, stage, deadline, rss_cap):
    fields(prior, ("path", "evidence_sha256", "raw_sha256"), "prior-fields")
    digest(prior["evidence_sha256"])
    digest(prior["raw_sha256"])
    raw_evidence, _ = read_pinned(prior["path"], EVIDENCE_CAP,
                                  deadline=deadline, rss_cap=rss_cap)
    need("sha256:" + hashlib.sha256(raw_evidence).hexdigest() ==
         prior["evidence_sha256"], "prior-evidence-drift")
    record = parse(raw_evidence, require_canonical=True)
    need(type(record) is dict and record.get("schema") == "e3-n1-small-metadata-v1"
         and record.get("stage") == stage and record.get("status") == "HOLD" and
         record.get("candidate") == request["candidate"], "prior-evidence-schema")
    checksum = record.get("checksum_sha256")
    digest(checksum)
    body = {key: value for key, value in record.items() if key != "checksum_sha256"}
    need(checksum == "sha256:" + hashlib.sha256(canonical(body)).hexdigest(),
         "prior-evidence-checksum")
    try:
        need(type(record["raw_base64"]) is str, "prior-raw-encoding")
        raw = base64.b64decode(record["raw_base64"], validate=True)
    except (KeyError, ValueError, TypeError, binascii.Error) as error:
        raise Hold("prior-raw-encoding") from error
    need(len(raw) <= JSON_CAP and
         "sha256:" + hashlib.sha256(raw).hexdigest() == prior["raw_sha256"] ==
         record.get("raw_sha256"), "prior-raw-drift")
    need(base64.b64encode(raw).decode("ascii") == record["raw_base64"],
         "prior-raw-noncanonical")
    need(time.monotonic() <= deadline and current_rss_bytes() <= rss_cap,
         "prior-resource-cap")
    return parse(raw)


def acquire(raw_request, reviewed_pin, test_only_index_digest=None):
    request_digest = "sha256:" + hashlib.sha256(raw_request).hexdigest()
    need(type(reviewed_pin) is str and reviewed_pin == request_digest,
         "unreviewed-request")
    if test_only_index_digest is None:
        need(REVIEWED_REQUEST_SHA256 is not None and
             reviewed_pin == REVIEWED_REQUEST_SHA256, "production-request-pin-absent")
    else:
        digest(test_only_index_digest)
        need(test_only_index_digest != F32_INDEX, "synthetic-override-real-index")
    request = parse(raw_request, require_canonical=True)
    fields(request, ("schema", "candidate", "stage", "source", "prior",
                     "selection", "limits"), "request-fields")
    need(request["schema"] == "e3-n1-small-request-v1", "request-schema")
    fields(request["candidate"], ("name", "index_digest"), "candidate-fields")
    expected_index = test_only_index_digest or F32_INDEX
    need(request["candidate"] == {"name": F32_NAME, "index_digest": expected_index},
         "candidate-identity")
    stage = request["stage"]
    need(stage in ("index", "manifest", "config"), "stage")
    fields(request["limits"], ("max_bytes", "max_seconds", "max_rss_bytes"),
           "limits-fields")
    limits = request["limits"]
    positive(limits["max_bytes"], JSON_CAP, "byte-limit")
    positive(limits["max_seconds"], 60, "time-limit")
    positive(limits["max_rss_bytes"], 2 * 1024**3, "rss-limit")
    deadline = time.monotonic() + limits["max_seconds"]
    source = request["source"]
    fields(source, ("path", "expected_digest", "expected_size", "identity"),
           "source-fields")
    absolute(source["path"])
    digest(source["expected_digest"])
    positive(source["expected_size"], limits["max_bytes"], "source-size")
    if stage == "index":
        need(request["prior"] is None and request["selection"] is None and
             source["expected_digest"] == expected_index, "index-dependency")
        prior = None
    else:
        need(request["prior"] is not None and request["selection"] is not None,
             "missing-prior-review")
        prior = prior_bytes(request["prior"], request,
                            "index" if stage == "manifest" else "manifest",
                            deadline, limits["max_rss_bytes"])
        if stage == "manifest":
            need(request["prior"]["raw_sha256"] == expected_index,
                 "prior-index-identity")
        if stage == "manifest":
            expected = selected_manifest(prior)
        else:
            expected, _ = manifest_config(prior)
        need(request["selection"] == expected and
             source["expected_digest"] == expected["digest"] and
             source["expected_size"] == expected["size"], "reviewed-selection-drift")
    raw, observed = read_pinned(source["path"], limits["max_bytes"],
                                source["identity"], deadline,
                                limits["max_rss_bytes"])
    need(len(raw) == source["expected_size"] and
         "sha256:" + hashlib.sha256(raw).hexdigest() == source["expected_digest"],
         "small-blob-drift")
    parsed = parse(raw)
    if stage == "index":
        selected = selected_manifest(parsed)
        claims = {"selected_manifest": selected}
    elif stage == "manifest":
        config, layers = manifest_config(parsed)
        claims = {"config": config, "ordered_layers": layers}
    else:
        _, layers = manifest_config(prior)
        diff_ids = config_claims(parsed, len(layers))
        need(all(layer["mediaType"] != TAR_TYPE or layer["digest"] == diff_id
                 for layer, diff_id in zip(layers, diff_ids)), "raw-tar-diffid")
        claims = {"diff_ids": diff_ids}
    payload = {"schema": "e3-n1-small-metadata-v1", "status": "HOLD",
               "stage": stage, "candidate": request["candidate"],
               "request_sha256": "sha256:" + hashlib.sha256(raw_request).hexdigest(),
               "raw_sha256": source["expected_digest"],
               "raw_size": len(raw), "raw_base64": base64.b64encode(raw).decode("ascii"),
               "source_path": source["path"], "read_time_identity": observed,
               "claims": claims,
               "limits": limits,
               "scope": "small-json-read-only;no-layer-or-custody-authority"}
    payload["checksum_sha256"] = "sha256:" + hashlib.sha256(canonical(payload)).hexdigest()
    result = canonical(payload)
    need(len(result) <= EVIDENCE_CAP, "evidence-byte-cap")
    need(time.monotonic() <= deadline and
         current_rss_bytes() <= limits["max_rss_bytes"], "stage-resource-cap")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    args = parser.parse_args()
    try:
        need(REVIEWED_REQUEST_SHA256 is not None, "production-request-pin-absent")
        raw, _ = read_pinned(args.request, REQUEST_CAP)
        result = acquire(raw, REVIEWED_REQUEST_SHA256)
    except (Hold, OSError) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
