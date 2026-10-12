"""Generated tiny fixtures for source-only v2 custody denial boundaries."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e3_private"))
import e3_n1_broker as broker
import e3_n1_durable as durable
from test_reconcile import fixture, seal


def synthetic_case(root):
    original = fixture(root)
    store = root / "out-of-export-receipts"
    store.mkdir(mode=0o700)
    release = durable.release_descriptor(
        original, receipt_store=store, image_id="synthetic:image-only",
        compiler_head="synthetic:compiler", compiler_tree="synthetic:compiler-tree",
        source_head="synthetic:source", source_tree="synthetic:source-tree")
    config = replace(original, pin=durable.release_pin(release))
    issued = broker.Broker(config).attempt(config.pin, config.target)
    assert issued.state == "VERIFIED"
    return config, release, store, issued


class DurableTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config, self.release, self.store, self.issued = synthetic_case(root)
        self.seal_dir = root / "external-seal-custody"
        self.seal_dir.mkdir(mode=0o700)
        self.token = None

    def external_seal(self, receipt=None):
        path = self.seal_dir / "seal.json"
        path.write_bytes(seal(receipt or self.issued.receipt))
        info = path.stat()
        return durable.SealRef(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                               info.st_dev, info.st_ino)

    def check(self, receipt=None, seal_ref=None, config=None, release=None):
        return durable.reconcile_durable(
            config or self.config, release or self.release, self.issued.attempt,
            self.store, receipt, self.token, seal_ref)

    def test_durable_success_requires_external_receipt_and_seal(self):
        result = durable.issue(self.config, self.release, self.issued, self.store)
        self.assertEqual(result.state, "DURABLE")
        self.token = result.token
        self.assertEqual(self.check().reason, "external-receipt-absent")
        self.assertEqual(self.check(self.issued.receipt).reason, "seal-absent")
        self.assertEqual(self.check(self.issued.receipt,
                                    self.external_seal()).state, "MATCHED")
        self.assertEqual(durable.issue(self.config, self.release, self.issued,
                                       self.store).state, "HOLD")
        self.assertEqual(self.check(b"{}", self.external_seal()).reason,
                         "external-receipt-mismatch")

    def test_crashes_before_commit_and_return_never_recover_without_external_input(self):
        labels = ("write:receipt-", "sync-file:receipt-", "publish:receipt-",
                  "sync-dir:receipt-", "write:commit-", "sync-file:commit-",
                  "publish:commit-", "sync-dir:commit-", "before-issue-return")
        preexisting_seal = self.external_seal()
        for label in labels:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                config, release, store, issued = synthetic_case(Path(temp))
                seal_path = Path(temp) / "preexisting-seal.json"
                seal_path.write_bytes(seal(issued.receipt))
                info = seal_path.stat()
                preexisting_seal = durable.SealRef(seal_path,
                    hashlib.sha256(seal_path.read_bytes()).hexdigest(),
                    info.st_dev, info.st_ino)
                pid = os.fork()
                if pid == 0:
                    def crash(name):
                        if name.startswith(label):
                            os._exit(77)
                    durable.issue(config, release, issued, store, crash)
                    os._exit(78)
                _, status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 77)
                result = durable.reconcile_durable(config, release,
                    issued.attempt, store, None, None, None)
                self.assertEqual(result.state, "HOLD")
                # Even an external receipt cannot compensate for a missing seal.
                self.assertEqual(durable.reconcile_durable(config, release,
                    issued.attempt, store, issued.receipt, None,
                    preexisting_seal).state, "HOLD")

    def test_faults_and_replacement_deny(self):
        for label in ("write:receipt-", "sync-file:receipt-", "publish:receipt-",
                      "sync-dir:receipt-", "write:commit-", "sync-file:commit-",
                      "publish:commit-", "sync-dir:commit-"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                config, release, store, issued = synthetic_case(Path(temp))
                seal_path = Path(temp) / "preexisting-seal.json"
                seal_path.write_bytes(seal(issued.receipt))
                info = seal_path.stat()
                valid_seal = durable.SealRef(seal_path,
                    hashlib.sha256(seal_path.read_bytes()).hexdigest(),
                    info.st_dev, info.st_ino)
                def fail(name):
                    if name.startswith(label):
                        raise OSError("injected")
                self.assertEqual(durable.issue(config, release,
                    issued, store, fail).state, "HOLD")
                self.assertEqual(durable.reconcile_durable(config, release,
                    issued.attempt, store, issued.receipt, None,
                    valid_seal).state, "HOLD")
        result = durable.issue(self.config, self.release, self.issued, self.store)
        self.assertEqual(result.state, "DURABLE")
        self.token = result.token
        good = self.external_seal()
        self.assertEqual(self.check(self.issued.receipt, good).state, "MATCHED")
        good.path.write_bytes(good.path.read_bytes() + b"x")
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        good = self.external_seal()
        self.token = b"wrong-token".ljust(32, b"x")
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        self.token = result.token
        receipt = self.store / ("receipt-" + self.issued.attempt + ".json")
        saved = receipt.read_bytes()
        receipt.write_bytes(b"{}")
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        receipt.write_bytes(saved)
        os.replace(receipt, self.store / "old-receipt")
        receipt.write_bytes(saved)
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        seal_path = good.path
        saved_seal = seal_path.read_bytes()
        os.replace(seal_path, self.seal_dir / "old-seal")
        seal_path.write_bytes(saved_seal)
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        seal_path.unlink()
        seal_path.symlink_to(self.seal_dir / "old-seal")
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")

    def test_release_and_target_drift_deny(self):
        result = durable.issue(self.config, self.release, self.issued, self.store)
        self.assertEqual(result.state, "DURABLE")
        self.token = result.token
        good = self.external_seal()
        self.assertEqual(self.check(self.issued.receipt, good).state, "MATCHED")
        changed = dict(self.release, image_id="synthetic:other-image")
        self.assertEqual(self.check(self.issued.receipt, good, release=changed).state,
                         "HOLD")
        changed_chain = dict(self.release, parent_chain=[[0, 0, 0]])
        self.assertEqual(self.check(self.issued.receipt, good,
                                    release=changed_chain).state, "HOLD")
        self.assertEqual(durable.issue(self.config, changed, self.issued,
                                       self.store).state, "HOLD")
        (self.config.parent / self.config.target / "index.json").write_bytes(b"drift")
        self.assertEqual(self.check(self.issued.receipt, good).state, "HOLD")
        self.assertEqual(self.check(self.issued.receipt, good).reason,
                         "target-files-mismatch")

    def test_foreign_store_duplicate_target_and_non_synthetic_release_deny(self):
        result = durable.issue(self.config, self.release, self.issued, self.store)
        self.assertEqual(result.state, "DURABLE")
        good = self.external_seal()
        token = json.loads((self.store / ("commit-" + self.issued.attempt +
                                          ".json")).read_bytes())
        self.assertEqual(token["schema"], "eshkol-e3-n1-receipt-commit-v2")
        self.assertEqual(durable.reconcile_durable(self.config, self.release,
            self.issued.attempt, self.seal_dir, self.issued.receipt,
            result.token, good).state, "HOLD")
        clone = self.store.parent / "cloned-receipts"
        shutil.copytree(self.store, clone)
        copied = clone / ("receipt-" + self.issued.attempt + ".json")
        copied_info = copied.stat()
        commit = clone / ("commit-" + self.issued.attempt + ".json")
        copied_commit = json.loads(commit.read_bytes())
        copied_commit["device"] = copied_info.st_dev
        copied_commit["inode"] = copied_info.st_ino
        commit.write_bytes(broker._encoded(copied_commit))
        self.assertEqual(durable.reconcile_durable(self.config, self.release,
            self.issued.attempt, clone, self.issued.receipt,
            result.token, good).state, "HOLD")
        moved = self.store.parent / "old-receipts"
        self.store.rename(moved)
        shutil.copytree(clone, self.store)
        self.assertEqual(durable.reconcile_durable(self.config, self.release,
            self.issued.attempt, self.store, self.issued.receipt,
            result.token, good).state, "HOLD")
        shutil.copytree(self.config.parent / self.config.target,
                        self.config.parent / "duplicate")
        with tempfile.TemporaryDirectory() as another:
            self.assertEqual(durable.issue(self.config, self.release, self.issued,
                Path(another)).state, "HOLD")
        with self.assertRaises(ValueError):
            durable.release_descriptor(self.config, receipt_store=self.store,
                image_id="sha256:unreviewed",
                compiler_head="synthetic:compiler", compiler_tree="synthetic:compiler-tree",
                source_head="synthetic:source", source_tree="synthetic:source-tree")

    def test_token_from_other_attempt_cannot_replay(self):
        first = durable.issue(self.config, self.release, self.issued, self.store)
        self.assertEqual(first.state, "DURABLE")
        second_base = replace(self.config, target="export2")
        second_store = self.store.parent / "other-receipts"
        second_store.mkdir()
        second_release = durable.release_descriptor(second_base, receipt_store=second_store,
            image_id="synthetic:image-only", compiler_head="synthetic:compiler",
            compiler_tree="synthetic:compiler-tree", source_head="synthetic:source",
            source_tree="synthetic:source-tree")
        second_config = replace(second_base, pin=durable.release_pin(second_release))
        second_outcome = broker.Broker(second_config).attempt(second_config.pin,
                                                               second_config.target)
        self.assertEqual(second_outcome.state, "VERIFIED")
        second = durable.issue(second_config, second_release, second_outcome,
                               second_store)
        self.assertEqual(second.state, "DURABLE")
        other_seal = self.seal_dir / "other-seal.json"
        other_seal.write_bytes(seal(second_outcome.receipt))
        info = other_seal.stat()
        seal_ref = durable.SealRef(other_seal,
            hashlib.sha256(other_seal.read_bytes()).hexdigest(), info.st_dev, info.st_ino)
        self.assertEqual(durable.reconcile_durable(second_config, second_release,
            second_outcome.attempt, second_store, second_outcome.receipt,
            first.token, seal_ref).state, "HOLD")
        self.assertEqual(durable.reconcile_durable(second_config, second_release,
            second_outcome.attempt, second_store, second_outcome.receipt,
            second.token, seal_ref).state, "MATCHED")


if __name__ == "__main__":
    unittest.main()
