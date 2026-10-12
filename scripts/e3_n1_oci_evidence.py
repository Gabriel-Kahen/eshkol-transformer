#!/usr/bin/env python3
"""Read-only, source-only OCI byte evidence. Never grants N1 admission."""

import argparse
import hashlib
import json
import os
import resource
import stat
import sys
import time
import zlib

CHUNK = 64 * 1024
JSON_CAP = 2 * 1024 * 1024
CATEGORIES = ("os", "tools", "packages", "headers", "paths", "environment")
INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
TAR_TYPE = "application/vnd.oci.image.layer.v1.tar"
GZIP_TYPE = TAR_TYPE + "+gzip"
CEILINGS = {"max_blob_bytes": 512 * 1024**2,
            "max_expanded_layer_bytes": 8 * 1024**3,
            "max_total_expanded_bytes": 16 * 1024**3,
            "max_evidence_bytes": 16 * 1024**2,
            "max_seconds": 1800,
            "max_rss_bytes": 2 * 1024**3}


class Hold(Exception):
    pass


def need(ok, reason):
    if not ok:
        raise Hold(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def parse_json(raw, canonical_required=False):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate-json-key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: need(False, "nonfinite-json"))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise Hold("invalid-json") from error
    if canonical_required:
        need(raw == canonical(value), "noncanonical-input")
    return value


def fields(value, names, reason):
    need(type(value) is dict and set(value) == set(names), reason)


def digest(value):
    need(type(value) is str and len(value) == 71 and value.startswith("sha256:")
         and all(ch in "0123456789abcdef" for ch in value[7:]), "bad-digest")
    return value[7:]


def positive(value, ceiling, reason):
    need(type(value) is int and 0 < value <= ceiling, reason)


def open_absolute(path, directory=False):
    need(type(path) is str and path.startswith("/") and "\x00" not in path,
         "nonabsolute-path")
    parts = path.split("/")[1:]
    need(bool(parts) and all(part not in ("", ".", "..") for part in parts),
         "noncanonical-path")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open("/", flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        result = os.open(parts[-1], flags | (os.O_DIRECTORY if directory else 0),
                         dir_fd=fd)
        mode = os.fstat(result).st_mode
        need(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), "wrong-file-kind")
        return result
    finally:
        os.close(fd)


def open_child(parent, parts, directory=False):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.dup(parent)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        result = os.open(parts[-1], flags | (os.O_DIRECTORY if directory else 0),
                         dir_fd=fd)
        mode = os.fstat(result).st_mode
        need(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), "wrong-file-kind")
        return result
    finally:
        os.close(fd)


def check_budget(deadline, rss_cap):
    need(time.monotonic() <= deadline, "time-cap")
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= rss_cap, "rss-cap")


def read_bytes(fd, cap, deadline, rss_cap):
    size = os.fstat(fd).st_size
    need(0 <= size <= cap, "byte-cap")
    parts = []
    count = 0
    while block := os.read(fd, CHUNK):
        check_budget(deadline, rss_cap)
        count += len(block)
        need(count <= cap, "byte-cap")
        parts.append(block)
    need(count == size, "file-size-drift")
    check_budget(deadline, rss_cap)
    return b"".join(parts)


def read_child_json(root, parts, deadline, rss_cap):
    fd = open_child(root, parts)
    try:
        raw = read_bytes(fd, JSON_CAP, deadline, rss_cap)
    finally:
        os.close(fd)
    return raw, parse_json(raw)


def descriptor(value, media_types):
    need(type(value) is dict and set(value) >= {"mediaType", "digest", "size"},
         "bad-descriptor")
    need(type(value["mediaType"]) is str and value["mediaType"] in media_types,
         "unsupported-media-type")
    digest(value["digest"])
    positive(value["size"], CEILINGS["max_blob_bytes"], "bad-descriptor-size")


def blob_fd(root, desc):
    return open_child(root, ("blobs", "sha256", digest(desc["digest"])))


def verified_json_blob(root, desc, deadline, rss_cap):
    fd = blob_fd(root, desc)
    try:
        raw = read_bytes(fd, JSON_CAP, deadline, rss_cap)
    finally:
        os.close(fd)
    need(len(raw) == desc["size"] and
         hashlib.sha256(raw).hexdigest() == digest(desc["digest"]), "blob-drift")
    return parse_json(raw)


def verified_layer(root, desc, diff_id, limits, deadline):
    fd = blob_fd(root, desc)
    compressed = hashlib.sha256()
    expanded = hashlib.sha256()
    compressed_size = expanded_size = 0
    decoder = zlib.decompressobj(31) if desc["mediaType"] == GZIP_TYPE else None
    try:
        need(os.fstat(fd).st_size == desc["size"], "layer-size-drift")
        while block := os.read(fd, CHUNK):
            check_budget(deadline, limits["max_rss_bytes"])
            compressed_size += len(block)
            need(compressed_size <= limits["max_blob_bytes"] and
                 compressed_size <= desc["size"], "compressed-byte-cap")
            compressed.update(block)
            if decoder is None:
                expanded_size += len(block)
                need(expanded_size <= limits["max_expanded_layer_bytes"],
                     "expanded-byte-cap")
                expanded.update(block)
            else:
                need(not decoder.eof, "gzip-trailing")
                pending = block
                while True:
                    try:
                        output = decoder.decompress(pending, CHUNK)
                    except zlib.error as error:
                        raise Hold("gzip-corrupt") from error
                    need(not decoder.unused_data, "gzip-trailing")
                    if output:
                        expanded_size += len(output)
                        need(expanded_size <= limits["max_expanded_layer_bytes"],
                             "expanded-byte-cap")
                        expanded.update(output)
                    tail = decoder.unconsumed_tail
                    check_budget(deadline, limits["max_rss_bytes"])
                    if decoder.eof:
                        need(not tail, "gzip-trailing")
                        break
                    if tail:
                        need(tail != pending or bool(output), "gzip-stalled")
                        pending = tail
                    elif len(output) == CHUNK:
                        pending = b""
                    else:
                        break
            check_budget(deadline, limits["max_rss_bytes"])
        need(decoder is None or decoder.eof, "gzip-short")
        need(compressed_size == desc["size"] and
             compressed.hexdigest() == digest(desc["digest"]), "compressed-drift")
        need(expanded.hexdigest() == digest(diff_id), "diffid-drift")
        return expanded_size
    finally:
        os.close(fd)


def validate_observations(observations, limits, deadline):
    fields(observations, CATEGORIES, "observation-categories")
    for category in CATEGORIES:
        group = observations[category]
        fields(group, ("state", "records"), "observation-group")
        need(group["state"] in ("UNMEASURED/HOLD", "external-unverified") and
             type(group["records"]) is list, "observation-state")
        need((group["state"] == "UNMEASURED/HOLD") == (len(group["records"]) == 0),
             "observation-presence")
        seen = set()
        for record in group["records"]:
            fields(record, ("subject", "value", "evidence"), "observation-record")
            for key in ("subject", "value"):
                need(type(record[key]) is str and 0 < len(record[key]) <= 4096,
                     "observation-text")
            need(record["subject"] not in seen, "duplicate-observation")
            seen.add(record["subject"])
            source = record["evidence"]
            fields(source, ("path", "size", "sha256", "producer", "method"),
                   "observation-source")
            digest(source["sha256"])
            positive(source["size"], limits["max_evidence_bytes"], "evidence-size")
            for key in ("producer", "method"):
                need(type(source[key]) is str and 0 < len(source[key]) <= 256,
                     "evidence-provenance")
            fd = open_absolute(source["path"])
            try:
                raw = read_bytes(fd, limits["max_evidence_bytes"], deadline,
                                 limits["max_rss_bytes"])
            finally:
                os.close(fd)
            need(len(raw) == source["size"] and
                 hashlib.sha256(raw).hexdigest() == digest(source["sha256"]),
                 "evidence-drift")


def verify(input_path):
    fd = open_absolute(input_path)
    try:
        raw_input = read_bytes(fd, JSON_CAP, time.monotonic() + 1800, 2 * 1024**3)
    finally:
        os.close(fd)
    plan = parse_json(raw_input, canonical_required=True)
    fields(plan, ("schema", "layout", "index_digest", "platform", "limits",
                  "observations"), "input-fields")
    need(plan["schema"] == "e3-n1-oci-input-v1", "input-schema")
    digest(plan["index_digest"])
    fields(plan["platform"], ("os", "architecture"), "platform-fields")
    need(all(type(plan["platform"][key]) is str and
             0 < len(plan["platform"][key]) <= 128 for key in ("os", "architecture")),
         "platform-value")
    fields(plan["limits"], CEILINGS, "limits-fields")
    limits = plan["limits"]
    for key, cap in CEILINGS.items():
        positive(limits[key], cap, "invalid-limit:" + key)
    deadline = time.monotonic() + limits["max_seconds"]
    root = open_absolute(plan["layout"], directory=True)
    try:
        _, layout = read_child_json(root, ("oci-layout",), deadline,
                                    limits["max_rss_bytes"])
        need(layout == {"imageLayoutVersion": "1.0.0"}, "oci-layout-version")
        index_raw, index = read_child_json(root, ("index.json",), deadline,
                                           limits["max_rss_bytes"])
        need(hashlib.sha256(index_raw).hexdigest() == digest(plan["index_digest"]),
             "index-drift")
        need(type(index) is dict and type(index.get("schemaVersion")) is int and
             index["schemaVersion"] == 2 and index.get("mediaType") == INDEX_TYPE and
             type(index.get("manifests")) is list, "index-schema")
        selected = [item for item in index["manifests"] if type(item) is dict and
                    item.get("platform") == plan["platform"]]
        need(len(selected) == 1, "platform-selection")
        manifest_desc = selected[0]
        descriptor(manifest_desc, {MANIFEST_TYPE})
        manifest = verified_json_blob(root, manifest_desc, deadline,
                                      limits["max_rss_bytes"])
        need(type(manifest) is dict and type(manifest.get("schemaVersion")) is int and
             manifest["schemaVersion"] == 2 and
             manifest.get("mediaType") == MANIFEST_TYPE and
             type(manifest.get("layers")) is list and manifest["layers"],
             "manifest-schema")
        config_desc = manifest.get("config")
        descriptor(config_desc, {CONFIG_TYPE})
        config = verified_json_blob(root, config_desc, deadline,
                                    limits["max_rss_bytes"])
        need(type(config) is dict and config.get("os") == plan["platform"]["os"] and
             config.get("architecture") == plan["platform"]["architecture"],
             "config-platform")
        rootfs = config.get("rootfs")
        need(type(rootfs) is dict and rootfs.get("type") == "layers" and
             type(rootfs.get("diff_ids")) is list and
             len(rootfs["diff_ids"]) == len(manifest["layers"]), "rootfs-schema")
        diff_ids = rootfs["diff_ids"]
        need(all(digest(item) for item in diff_ids), "rootfs-diffids")
        total = 0
        layers = []
        for desc, diff_id in zip(manifest["layers"], diff_ids):
            descriptor(desc, {TAR_TYPE, GZIP_TYPE})
            need(desc["size"] <= limits["max_blob_bytes"], "blob-byte-cap")
            expanded_size = verified_layer(root, desc, diff_id, limits, deadline)
            total += expanded_size
            need(total <= limits["max_total_expanded_bytes"], "total-expanded-cap")
            layers.append({"descriptor": desc, "diff_id": diff_id,
                           "expanded_size": expanded_size})
    finally:
        os.close(root)
    validate_observations(plan["observations"], limits, deadline)
    chain_id = diff_ids[0]
    for next_id in diff_ids[1:]:
        chain_id = "sha256:" + hashlib.sha256(
            (chain_id + " " + next_id).encode("ascii")).hexdigest()
    payload = {"schema": "e3-n1-oci-evidence-v1", "status": "HOLD",
               "input_sha256": "sha256:" + hashlib.sha256(raw_input).hexdigest(),
               "oci": {"index_digest": plan["index_digest"],
                       "platform": plan["platform"], "manifest": manifest_desc,
                       "config": config_desc, "layers": layers,
                       "rootfs": {"type": "layers", "diff_ids": diff_ids,
                                  "chain_id": chain_id,
                                  "chain_id_semantics": "derived-from-verified-diffids-only"}},
               "observations": plan["observations"]}
    payload["checksum_sha256"] = "sha256:" + hashlib.sha256(canonical(payload)).hexdigest()
    return canonical(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", help="canonical source-only input JSON")
    args = parser.parse_args()
    try:
        result = verify(args.input_manifest)
    except (Hold, OSError, KeyError, IndexError, zlib.error) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
