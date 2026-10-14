#!/usr/bin/env python3
"""SOURCE/PREP-only f32 OCI export input checker; always HOLD."""

import argparse
import hashlib
import json
import os
import stat
import sys

CANDIDATE = "eshkol-f32-quantum-llvm21:20260925"
INDEX_DIGEST = "sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
TAR_TYPE = "application/vnd.oci.image.layer.v1.tar"
GZIP_TYPE = TAR_TYPE + "+gzip"
MAX_INPUT = 2 * 1024 * 1024
OCI_JSON_CAP = 2 * 1024 * 1024  # Matches e3_n1_oci_evidence.py JSON_CAP.
MAX_LAYERS = 32
MAX_BLOB = 512 * 1024**2
MAX_TOTAL = 512 * 1024**2
MAX_OUTPUT = 2 * 1024**3
MAX_SECONDS = 1800
MAX_RSS = 2 * 1024**3
TOP_SLOTS = ("oci.layout", "oci.index_size", "oci.manifest", "oci.config", "oci.layers",
             "oci.diff_ids", "source_pins", "custody", "limits")


class Hold(Exception):
    pass


def need(condition, reason):
    if not condition:
        raise Hold(reason)


def encoded(value):
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError, RecursionError) as error:
        raise Hold("noncanonical-value") from error


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate-key")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda _: need(False, "nonfinite"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise Hold("invalid-json") from error
    need(raw == encoded(value), "noncanonical-input")
    return value


def fields(value, names, reason):
    need(type(value) is dict and set(value) == set(names), reason)


def digest(value):
    need(type(value) is str and len(value) == 71 and value.startswith("sha256:")
         and all(ch in "0123456789abcdef" for ch in value[7:]), "digest")


def integer(value, upper, reason, allow_zero=False):
    need(type(value) is int and (0 if allow_zero else 1) <= value <= upper, reason)


def absolute(path):
    need(type(path) is str and path.startswith("/") and "\x00" not in path and
         (path == "/" or all(part not in ("", ".", "..")
                             for part in path.split("/")[1:])),
         "absolute-path")


def open_input_nofollow(path):
    absolute(path)
    parts = path.split("/")[1:]
    need(bool(parts), "input-path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent = os.open("/", flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        return os.open(parts[-1], flags, dir_fd=parent)
    finally:
        os.close(parent)


def basename(value):
    need(type(value) is str and 0 < len(value) <= 128 and value not in (".", "..")
         and "/" not in value and "\x00" not in value, "basename")


def descriptor(value, allowed):
    fields(value, ("mediaType", "digest", "size"), "descriptor-fields")
    need(type(value["mediaType"]) is str and value["mediaType"] in allowed,
         "descriptor-media")
    digest(value["digest"])
    integer(value["size"], MAX_BLOB, "descriptor-size")


def path_pin(value, kind):
    fields(value, ("path", "resolved", "ancestors", "uid", "gid", "dev", "ino",
                   "mode", "acl_evidence_sha256"), "path-pin-fields")
    absolute(value["path"])
    absolute(value["resolved"])
    need(value["resolved"] == value["path"], "symlinked-path-unreviewed")
    integer(value["uid"], 2**32 - 1, "uid", allow_zero=True)
    integer(value["gid"], 2**32 - 1, "gid", allow_zero=True)
    integer(value["dev"], 2**63 - 1, "dev", allow_zero=True)
    integer(value["ino"], 2**63 - 1, "ino")
    integer(value["mode"], 2**32 - 1, "mode")
    need(stat.S_ISREG(value["mode"]) if kind == "file"
         else stat.S_ISDIR(value["mode"]), "path-kind")
    digest(value["acl_evidence_sha256"])
    parts = value["path"].split("/")[1:]
    expected = ["/"]
    for part in parts[:-1]:
        expected.append((expected[-1].rstrip("/") + "/" + part))
    need(type(value["ancestors"]) is list and
         [item.get("path") if type(item) is dict else None
          for item in value["ancestors"]] == expected, "ancestor-chain")
    for ancestor in value["ancestors"]:
        fields(ancestor, ("path", "uid", "gid", "dev", "ino", "mode",
                          "acl_evidence_sha256"), "ancestor-fields")
        absolute(ancestor["path"])
        for key in ("uid", "gid", "dev"):
            integer(ancestor[key], 2**63 - 1, "ancestor-id", allow_zero=True)
        integer(ancestor["ino"], 2**63 - 1, "ancestor-ino")
        integer(ancestor["mode"], 2**32 - 1, "ancestor-mode")
        need(stat.S_ISDIR(ancestor["mode"]), "ancestor-kind")
        digest(ancestor["acl_evidence_sha256"])


def evidence_ref(value):
    fields(value, ("path", "sha256", "size"), "evidence-ref-fields")
    absolute(value["path"])
    digest(value["sha256"])
    integer(value["size"], MAX_INPUT, "evidence-ref-size")


def verifier_anchor(value, role):
    fields(value, ("role", "algorithm", "public_key_hex", "public_key_sha256",
                   "verifier_binary", "verifier_binary_expected_size",
                   "verifier_binary_expected_sha256", "trust_evidence"),
           "verifier-anchor-fields")
    need(value["role"] == role and value["algorithm"] == "PureEd25519-v3",
         "verifier-role-or-algorithm")
    key = value["public_key_hex"]
    need(type(key) is str and len(key) == 64 and
         all(ch in "0123456789abcdef" for ch in key), "public-key")
    need(value["public_key_sha256"] ==
         "sha256:" + hashlib.sha256(bytes.fromhex(key)).hexdigest(), "public-key-digest")
    path_pin(value["verifier_binary"], "file")
    integer(value["verifier_binary_expected_size"], MAX_BLOB,
            "verifier-binary-size")
    digest(value["verifier_binary_expected_sha256"])
    evidence_ref(value["trust_evidence"])


def validate_complete(plan):
    oci = plan["oci"]
    fields(oci["layout"], ("sha256", "size"), "layout-fields")
    digest(oci["layout"]["sha256"])
    integer(oci["layout"]["size"], OCI_JSON_CAP, "small-json-cap")
    integer(oci["index_size"], OCI_JSON_CAP, "small-json-cap")
    fields(oci["manifest"], ("descriptor", "platform"), "manifest-fields")
    descriptor(oci["manifest"]["descriptor"], {MANIFEST_TYPE})
    need(oci["manifest"]["descriptor"]["size"] <= OCI_JSON_CAP,
         "small-json-cap")
    need(oci["manifest"]["platform"] == {"os": "linux", "architecture": "amd64"},
         "platform")
    descriptor(oci["config"], {CONFIG_TYPE})
    need(oci["config"]["size"] <= OCI_JSON_CAP, "small-json-cap")
    need(type(oci["layers"]) is list and 0 < len(oci["layers"]) <= MAX_LAYERS,
         "layers")
    need(type(oci["diff_ids"]) is list and
         len(oci["diff_ids"]) == len(oci["layers"]), "diff-id-count")
    for layer, diff_id in zip(oci["layers"], oci["diff_ids"]):
        descriptor(layer, {TAR_TYPE, GZIP_TYPE})
        digest(diff_id)
        if layer["mediaType"] == TAR_TYPE:
            need(layer["digest"] == diff_id, "raw-tar-diff-id")
    expected = [("oci-layout", oci["layout"]["size"], oci["layout"]["sha256"]),
                ("index.json", oci["index_size"], plan["candidate"]["index_digest"])]
    for item in (oci["manifest"]["descriptor"], oci["config"], *oci["layers"]):
        expected.append(("blobs/sha256/" + item["digest"][7:],
                         item["size"], item["digest"]))
    need(len({relative for relative, _, _ in expected}) == len(expected),
         "duplicate-export-blob")
    sources = plan["source_pins"]
    need(type(sources) is list and len(sources) == len(expected), "source-count")
    source_paths = set()
    for source, (relative, size, expected_sha) in zip(sources, expected):
        fields(source, ("relative", "source", "lstat_size", "expected_sha256"),
               "source-fields")
        need(source["relative"] == relative, "source-order-or-name")
        need(source["expected_sha256"] == expected_sha, "source-expected-digest")
        path_pin(source["source"], "file")
        need(source["source"]["path"] not in source_paths, "duplicate-source-path")
        source_paths.add(source["source"]["path"])
        integer(source["lstat_size"], MAX_BLOB, "source-lstat-size")
        need(source["lstat_size"] == size, "source-size-vs-descriptor")
    limits = plan["limits"]
    fields(limits, ("max_blob_bytes", "max_combined_blob_bytes", "max_output_bytes",
                    "max_seconds", "max_rss_bytes", "measured_free_bytes"), "limits-fields")
    for key, upper in (("max_blob_bytes", MAX_BLOB),
                       ("max_combined_blob_bytes", MAX_TOTAL),
                       ("max_output_bytes", MAX_OUTPUT),
                       ("max_seconds", MAX_SECONDS), ("max_rss_bytes", MAX_RSS)):
        integer(limits[key], upper, "limit:" + key)
    integer(limits["measured_free_bytes"], 2**63 - 1, "free-space")
    need(max(size for _, size, _ in expected) <= limits["max_blob_bytes"],
         "blob-limit-below-descriptor")
    need(sum(size for _, size, _ in expected) <=
         limits["max_combined_blob_bytes"], "combined-limit-below-descriptors")
    need(sum(size for _, size, _ in expected) <= limits["max_output_bytes"],
         "output-limit-below-descriptors")
    need(limits["measured_free_bytes"] >= 2 * limits["max_output_bytes"],
         "free-space-below-reserve")
    custody = plan["custody"]
    fields(custody, ("caller_uid", "broker_uid", "sealer_uid", "consumer_uid",
                     "output_parent", "intent_store", "receipt_store", "seal_store",
                     "one_use_store", "immutable_target_parent", "target_basename",
                     "stage_policy", "one_use_policy", "immutable_handoff_method",
                     "receipt_verifier", "seal_verifier", "authority_evidence"),
           "custody-fields")
    for key in ("caller_uid", "broker_uid", "sealer_uid", "consumer_uid"):
        integer(custody[key], 2**32 - 1, "custody-uid", allow_zero=True)
    need(len({custody["caller_uid"], custody["broker_uid"],
              custody["sealer_uid"]}) == 3, "authority-uid-separation")
    need(custody["consumer_uid"] not in
         (custody["broker_uid"], custody["sealer_uid"]),
         "consumer-minting-uid")
    store_names = ("output_parent", "intent_store", "receipt_store",
                   "seal_store", "one_use_store", "immutable_target_parent")
    for key in store_names:
        path_pin(custody[key], "directory")
    need(len({custody[key]["path"] for key in store_names}) == len(store_names),
         "custody-store-collision")
    for key in ("output_parent", "intent_store", "receipt_store", "one_use_store"):
        need(custody[key]["uid"] == custody["broker_uid"], "broker-store-owner")
    for key in ("seal_store", "immutable_target_parent"):
        need(custody[key]["uid"] == custody["sealer_uid"], "sealer-store-owner")
    for key in store_names:
        pin = custody[key]
        need(not stat.S_IMODE(pin["mode"]) & 0o022, "writable-store")
        disallowed = {custody["caller_uid"]}
        disallowed.add(custody["sealer_uid"] if key in (
            "output_parent", "intent_store", "receipt_store", "one_use_store")
            else custody["broker_uid"])
        for ancestor in pin["ancestors"]:
            need(not stat.S_IMODE(ancestor["mode"]) & 0o022,
                 "writable-store-ancestor")
            need(ancestor["uid"] not in disallowed or
                 not stat.S_IMODE(ancestor["mode"]) & 0o200,
                 "cross-authority-ancestor")
    basename(custody["target_basename"])
    need(custody["stage_policy"] == "unique-broker-owned" and
         custody["one_use_policy"] == "durable-pre-stage-HOLD" and
         custody["immutable_handoff_method"] in ("immutable-snapshot", "protected-ownership"),
         "custody-policy")
    verifier_anchor(custody["receipt_verifier"], "broker-receipt")
    verifier_anchor(custody["seal_verifier"], "independent-seal")
    need(custody["receipt_verifier"]["public_key_hex"] !=
         custody["seal_verifier"]["public_key_hex"], "shared-verifier-key")
    checks = custody["authority_evidence"]
    fields(checks, ("uid_acl", "mount_namespace", "proc_fds", "immutable_handoff",
                    "one_use_protocol", "receipt_issuance", "seal_custody"),
           "authority-evidence-fields")
    for item in checks.values():
        evidence_ref(item)


def evaluate(raw):
    plan = parse(raw)
    fields(plan, ("schema", "candidate", "oci", "source_pins", "custody", "limits"),
           "top-fields")
    need(plan["schema"] == "e3-n1-export-input-v1", "schema")
    fields(plan["candidate"], ("name", "index_digest"), "candidate-fields")
    need(plan["candidate"] == {"name": CANDIDATE, "index_digest": INDEX_DIGEST},
         "candidate-identity")
    fields(plan["oci"], ("layout", "index_size", "manifest", "config", "layers", "diff_ids"),
           "oci-fields")
    missing = []
    for slot in TOP_SLOTS:
        root, _, field = slot.partition(".")
        value = plan[root][field] if field else plan[root]
        if value is None:
            missing.append(slot)
    need(not missing or len(missing) == len(TOP_SLOTS),
         "partial-plan-unreviewed")
    if not missing:
        validate_complete(plan)
    payload = {"schema": "e3-n1-export-input-assessment-v1", "status": "HOLD",
               "pre_read_structure": "INCOMPLETE" if missing else "READY_FOR_REVIEW",
               "missing": missing,
               "input_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
               "candidate": plan["candidate"],
               "scope": "metadata-and-supplied-claims-only;no-source-read"}
    payload["checksum_sha256"] = "sha256:" + hashlib.sha256(encoded(payload)).hexdigest()
    return encoded(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest")
    args = parser.parse_args()
    try:
        before = os.lstat(args.input_manifest)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_INPUT,
             "input-kind-or-size")
        fd = open_input_nofollow(args.input_manifest)
        try:
            opened = os.fstat(fd)
            identity = lambda info: (info.st_dev, info.st_ino, info.st_mode,
                                     info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            need(identity(before) == identity(opened), "input-open-drift")
            raw = os.read(fd, MAX_INPUT + 1)
            need(len(raw) <= MAX_INPUT and len(raw) == opened.st_size and
                 identity(opened) == identity(os.fstat(fd)) and
                 identity(opened) == identity(os.lstat(args.input_manifest)),
                 "input-size-or-identity-drift")
        finally:
            os.close(fd)
        result = evaluate(raw)
    except (Hold, OSError) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
