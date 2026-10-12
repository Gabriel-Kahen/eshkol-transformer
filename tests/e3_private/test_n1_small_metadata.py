"""Only tiny synthetic OCI JSON; no f32 content-store or layer read."""

import base64
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("e3_n1_small_metadata.py")
SPEC = importlib.util.spec_from_file_location("e3_n1_small_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
GZIP_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def identity(path):
    fd, chain = MODULE.open_nofollow(str(path))
    try:
        return {"file": MODULE.identity(os.fstat(fd)), "ancestors": chain}
    finally:
        os.close(fd)


class SmallMetadataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.config = {"os": "linux", "architecture": "amd64",
                       "rootfs": {"type": "layers", "diff_ids": [sha(b"expanded")]} }
        self.config_raw = raw(self.config)
        self.config_path = self.home / "config.json"
        self.config_path.write_bytes(self.config_raw)
        self.config_desc = {"mediaType": CONFIG_TYPE, "digest": sha(self.config_raw),
                            "size": len(self.config_raw)}
        self.layer_desc = {"mediaType": GZIP_TYPE, "digest": sha(b"unread layer"),
                           "size": len(b"unread layer")}
        self.manifest = {"schemaVersion": 2, "mediaType": MANIFEST_TYPE,
                         "config": self.config_desc, "layers": [self.layer_desc]}
        self.manifest_raw = raw(self.manifest)
        self.manifest_path = self.home / "manifest.json"
        self.manifest_path.write_bytes(self.manifest_raw)
        self.manifest_desc = {"mediaType": MANIFEST_TYPE,
                              "digest": sha(self.manifest_raw),
                              "size": len(self.manifest_raw),
                              "platform": {"os": "linux", "architecture": "amd64"}}
        self.index = {"schemaVersion": 2, "mediaType": INDEX_TYPE,
                      "manifests": [self.manifest_desc]}
        self.index_raw = raw(self.index)
        self.index_path = self.home / "index.json"
        self.index_path.write_bytes(self.index_raw)
        self.candidate = {"name": MODULE.F32_NAME, "index_digest": sha(self.index_raw)}
        self.limits = {"max_bytes": 10000, "max_seconds": 10,
                       "max_rss_bytes": 2 * 1024**3}

    def request(self, stage, path, data, prior=None, selection=None):
        return {"schema": "e3-n1-small-request-v1", "candidate": self.candidate,
                "stage": stage,
                "source": {"path": str(path), "expected_digest": sha(data),
                           "expected_size": len(data), "identity": identity(path)},
                "prior": prior, "selection": selection, "limits": self.limits}

    def acquire(self, request):
        bytes_ = raw(request)
        return MODULE.acquire(bytes_, sha(bytes_),
                              test_only_index_digest=self.candidate["index_digest"])

    def index_evidence(self):
        result = self.acquire(self.request("index", self.index_path, self.index_raw))
        path = self.home / "index.evidence.json"
        path.write_bytes(result)
        return result, {"path": str(path), "evidence_sha256": sha(result),
                        "raw_sha256": sha(self.index_raw)}

    def manifest_evidence(self):
        _, prior = self.index_evidence()
        request = self.request("manifest", self.manifest_path, self.manifest_raw,
                               prior, self.manifest_desc)
        result = self.acquire(request)
        path = self.home / "manifest.evidence.json"
        path.write_bytes(result)
        return result, {"path": str(path), "evidence_sha256": sha(result),
                        "raw_sha256": sha(self.manifest_raw)}

    def rewrite_prior(self, evidence, prior, change):
        record = json.loads(evidence)
        change(record)
        body = dict(record)
        body.pop("checksum_sha256")
        record["checksum_sha256"] = sha(MODULE.canonical(body))
        changed = MODULE.canonical(record)
        Path(prior["path"]).write_bytes(changed)
        prior["evidence_sha256"] = sha(changed)

    def assert_hold(self, request):
        with self.assertRaises((MODULE.Hold, OSError)):
            self.acquire(request)

    def test_three_explicit_stages_preserve_raw_bytes(self):
        index_evidence, index_prior = self.index_evidence()
        index_record = json.loads(index_evidence)
        self.assertEqual(index_record["status"], "HOLD")
        self.assertEqual(base64.b64decode(index_record["raw_base64"]), self.index_raw)
        body = dict(index_record)
        checksum = body.pop("checksum_sha256")
        self.assertEqual(checksum, sha(raw(body)))
        manifest_request = self.request("manifest", self.manifest_path,
                                        self.manifest_raw, index_prior,
                                        self.manifest_desc)
        manifest_evidence = self.acquire(manifest_request)
        manifest_path = self.home / "manifest.evidence.json"
        manifest_path.write_bytes(manifest_evidence)
        manifest_prior = {"path": str(manifest_path),
                          "evidence_sha256": sha(manifest_evidence),
                          "raw_sha256": sha(self.manifest_raw)}
        config_request = self.request("config", self.config_path, self.config_raw,
                                      manifest_prior, self.config_desc)
        config_evidence = json.loads(self.acquire(config_request))
        self.assertEqual(config_evidence["claims"]["diff_ids"],
                         self.config["rootfs"]["diff_ids"])
        self.assertEqual(config_evidence["status"], "HOLD")
        # No layer file exists in the fixture.
        self.assertFalse((self.home / self.layer_desc["digest"][7:]).exists())

    def test_public_cli_has_no_pin_and_cannot_read(self):
        result = subprocess.run([sys.executable, str(SCRIPT),
                                 str(self.home / "does-not-exist.json")],
                                capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"production-request-pin-absent", result.stderr)

    def test_index_identity_and_platform_negatives(self):
        request = self.request("index", self.index_path, self.index_raw)
        request["source"]["identity"]["file"]["ino"] += 1
        self.assert_hold(request)
        request = self.request("index", self.index_path, self.index_raw)
        request["source"]["expected_digest"] = sha(b"wrong")
        self.assert_hold(request)
        self.index["manifests"].append(copy.deepcopy(self.manifest_desc))
        duplicate = raw(self.index)
        self.index_path.write_bytes(duplicate)
        self.candidate["index_digest"] = sha(duplicate)
        request = self.request("index", self.index_path, duplicate)
        self.assert_hold(request)

    def test_prior_review_and_selection_negatives(self):
        _, prior = self.index_evidence()
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        request["prior"]["evidence_sha256"] = sha(b"wrong")
        self.assert_hold(request)
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        request["selection"]["digest"] = sha(b"wrong")
        self.assert_hold(request)
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, None, self.manifest_desc)
        self.assert_hold(request)

    def test_malformed_prior_base64_holds(self):
        evidence, prior = self.index_evidence()
        self.rewrite_prior(evidence, prior,
                           lambda record: record.update(raw_base64=["not", "base64"]))
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        self.assert_hold(request)

    def test_prior_raw_size_and_claims_must_match_raw_bytes(self):
        evidence, prior = self.index_evidence()
        self.rewrite_prior(evidence, prior,
                           lambda record: record.update(raw_size=1))
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        with self.assertRaisesRegex(MODULE.Hold, "prior-raw-size"):
            self.acquire(request)

        evidence, prior = self.index_evidence()
        self.rewrite_prior(evidence, prior, lambda record: record["claims"]["selected_manifest"].update(size=1))
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        with self.assertRaisesRegex(MODULE.Hold, "prior-index-claims"):
            self.acquire(request)

        evidence, prior = self.manifest_evidence()
        self.rewrite_prior(evidence, prior,
                           lambda record: record["claims"]["config"].update(size=1))
        request = self.request("config", self.config_path,
                               self.config_raw, prior, self.config_desc)
        with self.assertRaisesRegex(MODULE.Hold, "prior-manifest-claims"):
            self.acquire(request)

    def test_prior_schema_fields_and_types_hold(self):
        evidence, prior = self.index_evidence()
        self.rewrite_prior(evidence, prior,
                           lambda record: record.update(unexpected=True))
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        with self.assertRaisesRegex(MODULE.Hold, "prior-evidence-fields"):
            self.acquire(request)

        evidence, prior = self.index_evidence()
        self.rewrite_prior(evidence, prior,
                           lambda record: record["limits"].update(max_bytes=True))
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        with self.assertRaisesRegex(MODULE.Hold, "prior-byte-limit"):
            self.acquire(request)

    def test_symlink_and_config_shape_negatives(self):
        link = self.home / "link.json"
        link.symlink_to(self.index_path)
        request = self.request("index", self.index_path, self.index_raw)
        request["source"]["path"] = str(link)
        self.assert_hold(request)
        _, prior = self.manifest_evidence()
        bad_config = copy.deepcopy(self.config)
        bad_config["rootfs"]["diff_ids"] = []
        bad_raw = raw(bad_config)
        self.config_path.write_bytes(bad_raw)
        request = self.request("config", self.config_path, bad_raw,
                               prior, self.config_desc)
        self.assert_hold(request)

    def test_noncanonical_and_limits_hold(self):
        request = self.request("index", self.index_path, self.index_raw)
        not_canonical = json.dumps(request).encode()
        with self.assertRaises(MODULE.Hold):
            MODULE.acquire(not_canonical, sha(not_canonical),
                           test_only_index_digest=self.candidate["index_digest"])
        request["limits"]["max_bytes"] = 1
        self.assert_hold(request)

    def test_large_small_json_evidence_and_output_cap(self):
        self.index["annotations"] = {"large": "x" * 1_800_000}
        self.index_raw = raw(self.index)
        self.index_path.write_bytes(self.index_raw)
        self.candidate["index_digest"] = sha(self.index_raw)
        self.limits["max_bytes"] = MODULE.JSON_CAP
        evidence, prior = self.index_evidence()
        self.assertLess(len(evidence), MODULE.EVIDENCE_CAP)
        self.assertGreater(len(evidence), 2 * 1024 * 1024)
        request = self.request("manifest", self.manifest_path,
                               self.manifest_raw, prior, self.manifest_desc)
        self.assertEqual(json.loads(self.acquire(request))["stage"], "manifest")
        saved = MODULE.EVIDENCE_CAP
        MODULE.EVIDENCE_CAP = 100
        try:
            self.assert_hold(self.request("index", self.index_path, self.index_raw))
        finally:
            MODULE.EVIDENCE_CAP = saved

    def test_synthetic_override_cannot_name_real_f32_index(self):
        request = self.request("index", self.index_path, self.index_raw)
        bytes_ = raw(request)
        with self.assertRaises(MODULE.Hold):
            MODULE.acquire(bytes_, sha(bytes_), test_only_index_digest=MODULE.F32_INDEX)


if __name__ == "__main__":
    unittest.main()
