import hashlib
import hmac
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e3_n1_broker as broker
import e3_n1_reconcile as reconcile


SEAL_KEY = b"independent-synthetic-seal-key"
SEAL_KEY_ID = "synthetic-seal-v1"


def fixture(root):
    parent = root / "protected"
    parent.mkdir(mode=0o700)
    inputs = root / "inputs"
    inputs.mkdir()
    payloads = (("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                ("index.json", b'{"manifests":[]}'),
                ("blobs/sha256/" + hashlib.sha256(b"tiny layer").hexdigest(),
                 b"tiny layer"))
    sources = []
    for number, (relative, content) in enumerate(payloads):
        path = inputs / str(number)
        path.write_bytes(content)
        sources.append(broker.Source(path, relative, hashlib.sha256(content).hexdigest(),
                                     len(content), path.resolve(), broker._path_inodes(path)))
    return broker.Config(parent, "a" * 64, "export", tuple(sources),
                         4096, 5, 2**40, b"synthetic-broker-key",
                         SEAL_KEY_ID, SEAL_KEY)


def seal(receipt, key=SEAL_KEY, key_id=SEAL_KEY_ID):
    body = json.loads(receipt)["body"]
    signed = {"key_id": key_id, "body": body}
    return broker._encoded(dict(signed, hmac_sha256=hmac.digest(
        key, broker._encoded(signed), "sha256").hex()))


def attempt_id(parent):
    records = list(parent.glob(".attempt-*.json"))
    assert len(records) == 1, records
    return records[0].name[len(".attempt-"):-len(".json")]


class ReconcileTest(unittest.TestCase):
    def test_hard_exit_at_publication_boundaries_never_promotes(self):
        boundaries = ("sync-dir:.attempt-*", "mkdir-stage", "open-stage",
                      "write:HOLD.json", "sync-initial-parent",
                      "before-read:index.json", "sync-stage", "rename-target",
                      "sync-renamed-parent", "write:VERIFIED.json",
                      "sync-file:VERIFIED.json", "sync-dir:VERIFIED.json",
                      "sync-verified-stage", "sync-verified-parent",
                      "before-receipt")
        no_stage = {"sync-dir:.attempt-*", "mkdir-stage"}
        unmarked_stage = {"open-stage", "write:HOLD.json"}
        marked_stage = {"sync-initial-parent", "before-read:index.json",
                        "sync-stage", "rename-target"}
        unverified_target = {"sync-renamed-parent", "write:VERIFIED.json",
                             "sync-file:VERIFIED.json"}
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temp:
                config = fixture(Path(temp))
                pid = os.fork()
                if pid == 0:
                    def crash(name):
                        if name == boundary or (boundary.endswith("*") and
                                                name.startswith(boundary[:-1])):
                            os._exit(77)
                    broker.Broker(config, crash).attempt(config.pin, config.target)
                    os._exit(78)
                _, status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 77)
                attempt = attempt_id(config.parent)
                result = reconcile.reconcile(config, attempt)
                self.assertEqual(result.state, "HOLD")
                intent = json.loads((config.parent / (".attempt-" + attempt +
                                                       ".json")).read_bytes())["body"]
                self.assertEqual(intent, {"attempt": attempt, "pin": config.pin,
                                          "target": config.target,
                                          "stage": ".stage-" + attempt,
                                          "state": "HOLD", "reason": "in-progress"})
                stage = config.parent / (".stage-" + attempt)
                target = config.parent / config.target
                if boundary in no_stage:
                    self.assertFalse(stage.exists())
                    self.assertFalse(target.exists())
                    self.assertEqual(result.reason, "target-absent")
                elif boundary in unmarked_stage | marked_stage:
                    self.assertTrue(stage.is_dir())
                    self.assertFalse(target.exists())
                    self.assertGreater(stage.stat().st_ino, 0)
                    marker = stage / "HOLD.json"
                    self.assertEqual(marker.is_file(), boundary in marked_stage)
                    self.assertEqual(result.reason, "evidence-unreadable" if
                                     boundary in unmarked_stage else
                                     "stage-present" if boundary == "sync-initial-parent"
                                     else "stage-evidence-mismatch")
                else:
                    self.assertFalse(stage.exists())
                    self.assertTrue(target.is_dir())
                    self.assertGreater(target.stat().st_ino, 0)
                    verified = target / "VERIFIED.json"
                    self.assertEqual(verified.is_file(), boundary not in unverified_target)
                    if verified.is_file():
                        body = json.loads(verified.read_bytes())["body"]
                        self.assertEqual((body["device"], body["inode"]),
                                         (target.stat().st_dev, target.stat().st_ino))
                    self.assertEqual(result.reason, "evidence-unreadable" if
                                     boundary in unverified_target else "receipt-absent")
                    self.assertEqual(reconcile.reconcile(config, attempt, b"{}", b"{}").state,
                                     "HOLD")

    def test_receipt_seal_intent_and_live_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            config = fixture(Path(temp))
            issued = broker.Broker(config).attempt(config.pin, config.target)
            self.assertEqual(issued.state, "VERIFIED")
            good_seal = seal(issued.receipt)
            check = lambda receipt=None, signed=None: reconcile.reconcile(
                config, issued.attempt, receipt, signed)
            self.assertEqual(check().reason, "receipt-absent")
            self.assertEqual(check(issued.receipt).reason, "seal-absent")
            self.assertEqual(check(issued.receipt, good_seal).state, "MATCHED")
            self.assertEqual(check(issued.receipt,
                                   seal(issued.receipt, config.signing_key)).state, "HOLD")
            self.assertEqual(check(issued.receipt,
                                   seal(issued.receipt, key_id="other-seal")).state, "HOLD")
            self.assertEqual(reconcile.reconcile(
                replace(config, seal_verification_key=config.signing_key),
                issued.attempt, issued.receipt, good_seal).state, "HOLD")
            with self.assertRaises(TypeError):
                reconcile.reconcile(config, issued.attempt, issued.receipt,
                                    seal(issued.receipt, b"caller-chosen-key"),
                                    seal_key=b"caller-chosen-key")
            forged = json.loads(issued.receipt)
            forged["body"]["attempt"] = "f" * 32
            self.assertEqual(check(broker._encoded(forged), good_seal).state, "HOLD")
            (config.parent / "export" / "index.json").write_bytes(b"drift")
            self.assertEqual(check(issued.receipt, good_seal).reason,
                             "target-files-mismatch")

    def test_duplicate_target_missing_records_and_inode_drift(self):
        for damage in ("duplicate", "missing-intent", "missing-marker", "inode-drift"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                config = fixture(Path(temp))
                issued = broker.Broker(config).attempt(config.pin, config.target)
                target = config.parent / config.target
                if damage == "duplicate":
                    shutil.copytree(target, config.parent / "duplicate")
                elif damage == "missing-intent":
                    (config.parent / (".attempt-" + issued.attempt + ".json")).unlink()
                elif damage == "missing-marker":
                    (target / "VERIFIED.json").unlink()
                else:
                    os.rename(target, config.parent / "old-target")
                    shutil.copytree(config.parent / "old-target", target)
                result = reconcile.reconcile(config, issued.attempt, issued.receipt,
                                             seal(issued.receipt))
                self.assertEqual(result.state, "HOLD")
                if damage == "duplicate":
                    self.assertEqual(result.reason, "duplicate-target")

    def test_stage_checksum_and_missing_marker_are_hold(self):
        for damage in ("checksum", "missing-marker"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                config = fixture(Path(temp))
                config.sources[-1].path.write_bytes(b"wrong layer")
                outcome = broker.Broker(config).attempt(config.pin, config.target)
                stage = config.parent / (".stage-" + outcome.attempt)
                if damage == "checksum":
                    (stage / "oci-layout").write_bytes(b"drift")
                else:
                    (stage / "HOLD.json").unlink()
                result = reconcile.reconcile(config, outcome.attempt)
                self.assertEqual(result.state, "HOLD")
                self.assertEqual(result.reason, "stage-evidence-mismatch" if
                                 damage == "checksum" else "evidence-unreadable")

    def test_unexpected_target_file_and_invalid_attempt_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            config = fixture(Path(temp))
            issued = broker.Broker(config).attempt(config.pin, config.target)
            (config.parent / "export" / "extra").write_bytes(b"unexpected")
            self.assertEqual(reconcile.reconcile(config, issued.attempt,
                             issued.receipt, seal(issued.receipt)).reason,
                             "target-layout-mismatch")
            self.assertEqual(reconcile.reconcile(config, "../export").reason,
                             "invalid-attempt")


if __name__ == "__main__":
    unittest.main()
