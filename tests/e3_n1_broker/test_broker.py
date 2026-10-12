import hashlib
import hmac
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e3_n1_broker as broker


class BrokerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / "protected"
        self.parent.mkdir(mode=0o700)
        source_dir = self.root / "inputs"
        source_dir.mkdir()
        payloads = (("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                    ("index.json", b'{"manifests":[]}'),
                    ("blobs/sha256/" + hashlib.sha256(b"tiny layer").hexdigest(),
                     b"tiny layer"))
        sources = []
        for i, (relative, content) in enumerate(payloads):
            path = source_dir / str(i)
            path.write_bytes(content)
            sources.append(broker.Source(path, relative, hashlib.sha256(content).hexdigest(),
                                         len(content)))
        self.config = broker.Config(self.parent, "a" * 64, "export", tuple(sources),
                                    4096, 5, 2**40, b"broker-only-test-key")

    def run_attempt(self, fault=None):
        seen = []
        def checkpoint(name):
            seen.append(name)
            if name == fault:
                raise OSError("injected " + name)
        result = broker.Broker(self.config, checkpoint).attempt(self.config.pin,
                                                                 self.config.target)
        return result, seen

    def intent(self, attempt):
        fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return broker._read_record(fd, ".attempt-" + attempt + ".json")
        finally:
            os.close(fd)

    def seal(self, receipt):
        body = json.loads(receipt)["body"]
        return broker._encoded({"body": body, "hmac_sha256": hmac.digest(
            b"independent-seal-key", broker._encoded(body), "sha256").hex()})

    def verify(self, result, seal=None):
        return broker.verify_receipt(self.config, result.receipt or b"{}",
                                     seal or b"{}", b"independent-seal-key")

    def test_success_requires_both_authorities_and_live_identity(self):
        result, seen = self.run_attempt()
        self.assertEqual(result.state, "VERIFIED")
        self.assertLess(seen.index("sync-initial-parent"),
                        seen.index("before-read:blobs/sha256/" + self.config.sources[2].sha256))
        self.assertEqual(self.intent(result.attempt)["state"], "HOLD")
        self.assertFalse(self.verify(result))
        self.assertTrue(self.verify(result, self.seal(result.receipt)))
        (self.parent / "export" / "index.json").write_bytes(b"drift")
        self.assertFalse(self.verify(result, self.seal(result.receipt)))

    def test_intent_failure_has_no_stage_or_read(self):
        # Unique names preclude an exact static label; inject all intent syncs.
        def fail_intent(name):
            if name.startswith("sync-dir:.attempt-"):
                raise OSError("injected")
        result = broker.Broker(self.config, fail_intent).attempt(self.config.pin, "export")
        self.assertEqual(result.state, "HOLD")
        self.assertFalse(list(self.parent.glob(".stage-*")))
        self.assertFalse(self.verify(result))

    def test_initial_marker_and_sync_fail_closed(self):
        for fault in ("write:HOLD.json", "sync-file:HOLD.json", "sync-dir:HOLD.json",
                      "sync-initial-parent"):
            with self.subTest(fault=fault):
                result, seen = self.run_attempt(fault)
                self.assertEqual(result.state, "INDETERMINATE")
                self.assertTrue(self.intent(result.attempt)["state"] == "HOLD")
                self.assertFalse(any(s.startswith("before-read:") for s in seen))
                self.assertFalse(self.verify(result))

    def test_stage_open_checked_cleanup_and_cleanup_fault(self):
        result, seen = self.run_attempt("open-stage")
        self.assertEqual(result.state, "HOLD")
        self.assertFalse((self.parent / (".stage-" + result.attempt)).exists())
        self.assertFalse(any(s.startswith("before-read:") for s in seen))
        def fail(name):
            if name == "open-stage" or name == "sync-cleanup-parent":
                raise OSError("injected")
        other = broker.Broker(self.config, fail).attempt(self.config.pin, "export")
        self.assertEqual(other.state, "INDETERMINATE")
        self.assertEqual(self.intent(other.attempt)["state"], "HOLD")

    def test_wrong_hash_keeps_partial_and_hold(self):
        source = self.config.sources[2]
        source.path.write_bytes(b"evil layer")
        result, _ = self.run_attempt()
        self.assertEqual(result.state, "HOLD")
        stage = self.parent / (".stage-" + result.attempt)
        self.assertTrue((stage / source.relative).exists())
        self.assertEqual(json.loads((stage / "HOLD.json").read_bytes())["body"]["state"],
                         "HOLD")
        self.assertFalse(self.verify(result))

    def test_partial_marker_sync_fault_is_indeterminate(self):
        source = self.config.sources[2]
        source.path.write_bytes(b"evil layer")
        count = 0
        def fault(name):
            nonlocal count
            if name == "sync-file:HOLD.json":
                count += 1
                if count == 2:
                    raise OSError("injected final marker")
        result = broker.Broker(self.config, fault).attempt(self.config.pin, "export")
        self.assertEqual(result.state, "INDETERMINATE")
        self.assertEqual(self.intent(result.attempt)["state"], "HOLD")

    def test_duplicate_target_and_postrename_faults(self):
        (self.parent / "export").mkdir()
        result, _ = self.run_attempt()
        self.assertEqual(result.state, "HOLD")
        self.assertIsNone(result.receipt)
        self.assertTrue((self.parent / "export").is_dir())
        (self.parent / "export").rmdir()
        for fault in ("sync-renamed-parent", "write:VERIFIED.json",
                      "sync-verified-parent"):
            with self.subTest(fault=fault):
                result, _ = self.run_attempt(fault)
                self.assertIsNone(result.receipt)
                self.assertFalse(self.verify(result))
                self.assertFalse((self.parent / "export").exists())
                self.assertEqual(self.intent(result.attempt)["state"], "HOLD")

    def test_retraction_fault_quarantines_visible_target(self):
        def fault(name):
            if name in ("sync-verified-parent", "retract-target"):
                raise OSError("injected")
        result = broker.Broker(self.config, fault).attempt(self.config.pin, "export")
        self.assertEqual(result.state, "INDETERMINATE")
        self.assertTrue((self.parent / "export").exists())
        self.assertIsNone(result.receipt)
        self.assertFalse(self.verify(result))

    def test_stale_receipt_rejected_after_target_replacement(self):
        result, _ = self.run_attempt()
        seal = self.seal(result.receipt)
        self.assertTrue(self.verify(result, seal))
        os.rename(self.parent / "export", self.parent / "quarantine")
        (self.parent / "export").mkdir()
        self.assertFalse(self.verify(result, seal))

    def test_source_symlink_and_identity_drift_hold(self):
        original = self.config.sources[2].path
        replacement = original.with_name("replacement")
        replacement.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(replacement)
        result, _ = self.run_attempt()
        self.assertEqual(result.state, "HOLD")
        self.assertIsNone(result.receipt)
        original.unlink()
        original.write_bytes(replacement.read_bytes())
        def drift(name):
            if name == "read-chunk:" + self.config.sources[2].relative:
                original.write_bytes(b"evil layer")
        result = broker.Broker(self.config, drift).attempt(self.config.pin, "export")
        self.assertEqual(result.state, "HOLD")
        self.assertIsNone(result.receipt)

    def test_unreviewed_request_and_bad_seal(self):
        result = broker.Broker(self.config).attempt("b" * 64, "export")
        self.assertEqual((result.state, result.attempt), ("HOLD", ""))
        self.assertFalse(list(self.parent.iterdir()))
        result, _ = self.run_attempt()
        seal = bytearray(self.seal(result.receipt))
        seal[-3] ^= 1
        self.assertFalse(self.verify(result, bytes(seal)))
        self.assertFalse(self.verify(result, self.seal(result.receipt) + b"corrupt"))


if __name__ == "__main__":
    unittest.main()
