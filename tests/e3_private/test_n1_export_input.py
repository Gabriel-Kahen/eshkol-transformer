"""Synthetic metadata only: no OCI blob, export, broker, or Docker operation."""

import copy
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/e3_n1_export_input.py"
CANDIDATE_FILE = ROOT / "docs/e3/eshkol-f32-quantum-llvm21-20260925.export-input.json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
TAR = "application/vnd.oci.image.layer.v1.tar"
GZIP = TAR + "+gzip"


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def descriptor(media, data, size):
    return {"mediaType": media, "digest": sha(data), "size": size}


def pin(path, kind="directory", uid=2001):
    parts = path.split("/")[1:]
    ancestors = ["/"]
    for part in parts[:-1]:
        ancestors.append(ancestors[-1].rstrip("/") + "/" + part)
    acl = sha(b"synthetic acl evidence")
    base = {"uid": uid, "gid": uid, "dev": 1, "ino": 10,
            "mode": stat.S_IFDIR | 0o700, "acl_evidence_sha256": acl}
    return {**base, "path": path, "resolved": path,
            "mode": (stat.S_IFREG | 0o600) if kind == "file" else base["mode"],
            "ancestors": [{**base, "path": item, "ino": i + 1, "uid": 0, "gid": 0}
                          for i, item in enumerate(ancestors)]}


def evidence(name):
    return {"path": "/synthetic/evidence/" + name,
            "size": 1, "sha256": sha(name.encode())}


def anchor(role, byte):
    key = bytes([byte]) * 32
    return {"role": role, "algorithm": "PureEd25519-v3",
            "public_key_hex": key.hex(), "public_key_sha256": sha(key),
            "verifier_binary": pin("/synthetic/verifier/" + role, "file"),
            "verifier_binary_expected_size": 1,
            "verifier_binary_expected_sha256": sha(role.encode()),
            "trust_evidence": evidence(role)}


def complete_plan():
    plan = json.loads(CANDIDATE_FILE.read_bytes())
    manifest = descriptor(MANIFEST, b"synthetic manifest", 100)
    config = descriptor(CONFIG, b"synthetic config", 80)
    layers = [descriptor(GZIP, b"synthetic gzip 0", 40),
              descriptor(TAR, b"synthetic tar 1", 50)]
    plan["oci"] = {"index_size": 200,
                   "manifest": {"descriptor": manifest,
                                "platform": {"os": "linux", "architecture": "amd64"}},
                   "config": config, "layers": layers,
                   "diff_ids": [sha(b"synthetic expanded 0"), layers[1]["digest"]]}
    plan["oci"]["layout"] = {"size": 40, "sha256": sha(b"synthetic oci layout")}
    sources = [("oci-layout", 40, plan["oci"]["layout"]["sha256"]),
               ("index.json", 200, plan["candidate"]["index_digest"])]
    sources.extend(("blobs/sha256/" + d["digest"][7:], d["size"], d["digest"])
                   for d in (manifest, config, *layers))
    plan["source_pins"] = [
        {"relative": relative, "source": pin("/synthetic/source/" + str(i), "file"),
         "lstat_size": size, "expected_sha256": expected_sha}
        for i, (relative, size, expected_sha) in enumerate(sources)]
    plan["limits"] = {"max_blob_bytes": 1000, "max_combined_blob_bytes": 1000,
                      "max_output_bytes": 1000, "max_seconds": 10,
                      "max_rss_bytes": 100000000, "measured_free_bytes": 2000}
    stores = ("output_parent", "intent_store", "receipt_store", "seal_store",
              "one_use_store", "immutable_target_parent")
    custody = {key: pin("/synthetic/protected/" + key,
                        uid=2002 if key in ("seal_store", "immutable_target_parent") else 2001)
               for key in stores}
    custody.update({"caller_uid": 1000, "broker_uid": 2001, "sealer_uid": 2002,
                    "consumer_uid": 1000, "target_basename": "synthetic-target",
                    "stage_policy": "unique-broker-owned",
                    "one_use_policy": "durable-pre-stage-HOLD",
                    "immutable_handoff_method": "immutable-snapshot",
                    "receipt_verifier": anchor("broker-receipt", 1),
                    "seal_verifier": anchor("independent-seal", 2),
                    "authority_evidence": {
                        key: evidence(key) for key in (
                            "uid_acl", "mount_namespace", "proc_fds",
                            "immutable_handoff", "one_use_protocol",
                            "receipt_issuance", "seal_custody")}})
    plan["custody"] = custody
    return plan


class ExportInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.input = Path(temp.name) / "input.json"

    def run_plan(self, plan=None, data=None):
        self.input.write_bytes(data if data is not None else raw(plan))
        return subprocess.run([sys.executable, str(SCRIPT), str(self.input)],
                              capture_output=True, check=False)

    def assert_hold_error(self, plan=None, data=None):
        result = self.run_plan(plan, data)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"HOLD:", result.stderr)

    def test_real_candidate_is_explicitly_incomplete(self):
        result = self.run_plan(data=CANDIDATE_FILE.read_bytes())
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "HOLD")
        self.assertEqual(output["pre_read_structure"], "INCOMPLETE")
        self.assertEqual(len(output["missing"]), 9)
        self.assertEqual(output["candidate"]["index_digest"],
                         "sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d")

    def test_synthetic_complete_is_only_ready_for_review(self):
        result = self.run_plan(complete_plan())
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(result.stdout, raw(output))
        checksum = output.pop("checksum_sha256")
        self.assertEqual(checksum, sha(raw(output)))
        self.assertEqual(output["status"], "HOLD")
        self.assertEqual(output["pre_read_structure"], "READY_FOR_REVIEW")
        self.assertEqual(output["missing"], [])

    def test_source_and_descriptor_mutants_hold(self):
        mutations = []
        plan = complete_plan()
        plan["source_pins"][2]["lstat_size"] += 1
        mutations.append(plan)
        plan = complete_plan()
        plan["source_pins"][2], plan["source_pins"][3] = (
            plan["source_pins"][3], plan["source_pins"][2])
        mutations.append(plan)
        plan = complete_plan()
        plan["oci"]["layers"][1]["digest"] = sha(b"wrong raw tar")
        mutations.append(plan)
        plan = complete_plan()
        plan["oci"]["manifest"]["platform"]["architecture"] = "arm64"
        mutations.append(plan)
        plan = complete_plan()
        plan["source_pins"][0]["source"]["ancestors"][0]["mode"] = stat.S_IFREG | 0o600
        mutations.append(plan)
        plan = complete_plan()
        plan["source_pins"][0]["expected_sha256"] = sha(b"wrong layout")
        mutations.append(plan)
        plan = complete_plan()
        plan["oci"]["layout"]["size"] += 1
        mutations.append(plan)
        for case in mutations:
            with self.subTest(case=case):
                self.assert_hold_error(case)

    def test_authority_mutants_hold(self):
        mutations = []
        plan = complete_plan()
        plan["custody"]["sealer_uid"] = plan["custody"]["broker_uid"]
        mutations.append(plan)
        plan = complete_plan()
        plan["custody"]["seal_verifier"]["public_key_hex"] = (
            plan["custody"]["receipt_verifier"]["public_key_hex"])
        plan["custody"]["seal_verifier"]["public_key_sha256"] = (
            plan["custody"]["receipt_verifier"]["public_key_sha256"])
        mutations.append(plan)
        plan = complete_plan()
        plan["custody"]["seal_store"] = copy.deepcopy(plan["custody"]["receipt_store"])
        mutations.append(plan)
        plan = complete_plan()
        del plan["custody"]["authority_evidence"]["proc_fds"]
        mutations.append(plan)
        plan = complete_plan()
        plan["limits"]["measured_free_bytes"] = 1000
        mutations.append(plan)
        plan = complete_plan()
        plan["custody"]["receipt_verifier"]["verifier_binary_expected_sha256"] = "bad"
        mutations.append(plan)
        plan = complete_plan()
        plan["custody"]["seal_store"]["uid"] = plan["custody"]["broker_uid"]
        mutations.append(plan)
        for case in mutations:
            with self.subTest(case=case):
                self.assert_hold_error(case)

    def test_noncanonical_and_overflow_hold(self):
        plan = complete_plan()
        self.assert_hold_error(data=json.dumps(plan).encode())
        self.assert_hold_error(data=raw(plan).replace(b'"schema":',
                                                     b'"schema":"duplicate","schema":', 1))
        self.assert_hold_error(data=b'{"schema":1e999}\n')

    def test_input_symlink_holds(self):
        target = self.input.parent / "target.json"
        target.write_bytes(raw(complete_plan()))
        self.input.symlink_to(target)
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.input)],
                                capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"HOLD:", result.stderr)


if __name__ == "__main__":
    unittest.main()
