"""Synthetic SOURCE/PREP and one-use marker negatives; no build or real marker."""

import concurrent.futures
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().with_name("e3_n1_prebuild.py")
spec = importlib.util.spec_from_file_location("e3_n1_prebuild", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture_tar(path, names):
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": "a" * 40}) as archive:
        for name in names:
            body = b"synthetic"
            info = tarfile.TarInfo(name)
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))


class PrebuildTests(unittest.TestCase):
    def test_canonical_and_pins(self):
        source = {"archive": {"sha256": mod.SOURCE_SHA256},
                  "verification": {"primary_fingerprint": mod.PRIMARY_FINGERPRINT}}
        value = mod.candidate(source, SCRIPT.parents[2] / "toolchain/eshkol.lock")
        data = mod.canonical(value)
        self.assertEqual(mod.validate_candidate(data, mod.sha256(data)), value)
        with self.assertRaisesRegex(mod.Hold, "candidate-digest-drift"):
            mod.validate_candidate(data, "0" * 64)
        bad = json.loads(data)
        bad["base"]["tree"] = "0" * 40
        with self.assertRaisesRegex(mod.Hold, "base-or-schema-drift"):
            mod.validate_candidate(mod.canonical(bad))
        bad = json.loads(data)
        bad["source"]["archive"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(mod.Hold, "source-pin-drift"):
            mod.validate_candidate(mod.canonical(bad))
        bad = json.loads(data)
        bad["source"]["verification"]["primary_fingerprint"] = "0" * 40
        with self.assertRaisesRegex(mod.Hold, "trust-anchor-drift"):
            mod.validate_candidate(mod.canonical(bad))
        with self.assertRaisesRegex(mod.Hold, "duplicate-json-key"):
            mod.parse_canonical(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(mod.Hold, "noncanonical-json"):
            mod.parse_canonical(b'{"a": 1}\n')

    def test_archive_inventory_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tiny.tar.gz"
            fixture_tar(path, ["openssl-3.5.9/file"])
            with self.assertRaisesRegex(mod.Hold, "tar-root-or-pax"):
                mod.scan_archive(path)
            fixture_tar(path, ["openssl-3.5.9/", "openssl-3.5.9/file"])
            inventory = mod.scan_archive(path)
            self.assertEqual(len(inventory["members"]), 2)
            self.assertEqual(inventory["members"][1]["sha256"], mod.sha256(b"synthetic"))
            fixture_tar(path, ["openssl-3.5.9/", "openssl-3.5.9/../escape"])
            with self.assertRaisesRegex(mod.Hold, "tar-unsafe-path"):
                mod.scan_archive(path)
            fixture_tar(path, ["openssl-3.5.9/", "openssl-3.5.9/file",
                               "openssl-3.5.9/file"])
            with self.assertRaisesRegex(mod.Hold, "tar-duplicate-effective-path"):
                mod.scan_archive(path)

    def test_pax_and_link_rejected(self):
        row = b"52 comment=" + b"a" * 40 + b"\n"
        self.assertEqual(len(row), 52)
        with self.assertRaisesRegex(mod.Hold, "tar-duplicate-pax"):
            mod._pax(row + row)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "link.tar.gz"
            with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT,
                              pax_headers={"comment": "a" * 40}) as archive:
                root = tarfile.TarInfo("openssl-3.5.9/")
                root.type = tarfile.DIRTYPE
                archive.addfile(root)
                link = tarfile.TarInfo("openssl-3.5.9/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../escape"
                archive.addfile(link)
            with self.assertRaisesRegex(mod.Hold, "tar-link-or-unsupported-kind"):
                mod.scan_archive(path)

    def test_archive_swap_between_signature_and_scan_is_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "openssl-3.5.9.tar.gz"
            replacement = root / "replacement.tar.gz"
            fixture_tar(archive, ["openssl-3.5.9/", "openssl-3.5.9/first"])
            fixture_tar(replacement, ["openssl-3.5.9/", "openssl-3.5.9/second"])
            signature = root / "openssl-3.5.9.tar.gz.asc"
            certificate = root / "pubkeys.asc"
            signature.write_bytes(b"synthetic signature")
            certificate.write_bytes(b"synthetic certificate")
            def swap(snapshot):
                snapshot.write_bytes(replacement.read_bytes())
            with mock.patch.object(mod, "SOURCE_SHA256", mod.file_hash(archive)), \
                 mock.patch.object(mod, "verify_signature", return_value={"result": "synthetic"}):
                with self.assertRaisesRegex(mod.Hold, "source-changed-during-verification"):
                    mod.observed_source(archive, signature, certificate, root / "inventory.json",
                                        after_verify=swap)

    def test_no_replace_evidence_output(self):
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / "evidence"
            private.mkdir(mode=0o700)
            output = private / "inventory.json"
            output.write_bytes(b"sealed-old")
            with self.assertRaisesRegex(mod.Hold, "output-preexisting"):
                mod.publish_new(output, b"new")
            self.assertEqual(output.read_bytes(), b"sealed-old")
            target = Path(temp) / "unrelated"
            target.write_bytes(b"untouched")
            link = private / "candidate.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(mod.Hold, "output-preexisting"):
                mod.publish_new(link, b"new")
            self.assertEqual(target.read_bytes(), b"untouched")
            fresh = private / "fresh.json"
            mod.publish_new(fresh, b"canonical\n")
            self.assertEqual(fresh.read_bytes(), b"canonical\n")
            self.assertEqual(fresh.stat().st_mode & 0o777, 0o600)

    def test_prepared_pin_replay_and_marker_fail_closed(self):
        digest = "a" * 64
        nonce = "b" * 64
        record = mod.prepared_record(digest, nonce, digest)
        for changed in (dict(reviewed_digest="c" * 64),
                        dict(base_tree="0" * 40), dict(source_sha256="0" * 64)):
            args = {"reviewed_digest": digest, **changed}
            with self.assertRaisesRegex(mod.Hold, "prepared-pin-mismatch"):
                mod.prepared_record(digest, nonce, **args)
        with tempfile.TemporaryDirectory() as temp:
            marker_dir = Path(temp) / "protected"
            marker_dir.mkdir(mode=0o700)
            chain = mod.path_chain(marker_dir)
            with self.assertRaisesRegex(mod.Hold, "injected-after-write"):
                mod.consume_once(record, marker_dir, chain, fault="after-write")
            with self.assertRaisesRegex(mod.Hold, "marker-spent"):
                mod.consume_once(record, marker_dir, chain)
            self.assertTrue(list(marker_dir.iterdir()))
            other = mod.prepared_record(digest, "c" * 64, digest)
            marker_dir2 = Path(temp) / "protected2"
            marker_dir2.mkdir(mode=0o700)
            chain2 = mod.path_chain(marker_dir2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: self._consume(other, marker_dir2, chain2), range(2)))
            self.assertEqual(sorted(results), ["spent", "success"])
            with self.assertRaisesRegex(mod.Hold, "marker-ancestor-drift"):
                mod.consume_once(other, marker_dir2, chain)

    def test_each_uncertain_marker_boundary_stays_spent(self):
        digest = "d" * 64
        with tempfile.TemporaryDirectory() as temp:
            for index, fault in enumerate(("after-create", "after-write",
                                           "after-readback", "after-sync")):
                directory = Path(temp) / str(index)
                directory.mkdir(mode=0o700)
                chain = mod.path_chain(directory)
                record = mod.prepared_record(digest, f"{index:064x}", digest)
                with self.assertRaisesRegex(mod.Hold, "injected-" + fault):
                    mod.consume_once(record, directory, chain, fault=fault)
                with self.assertRaisesRegex(mod.Hold, "marker-spent"):
                    mod.consume_once(record, directory, chain)
            fresh = Path(temp) / "fresh"
            fresh.mkdir(mode=0o700)
            record = mod.prepared_record(digest, "f" * 64, digest)
            with self.assertRaisesRegex(mod.Hold, "injected-before-create"):
                mod.consume_once(record, fresh, mod.path_chain(fresh), fault="before-create")
            self.assertEqual(list(fresh.iterdir()), [])

    def test_hard_exit_after_marker_write_stays_spent(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "protected"
            directory.mkdir(mode=0o700)
            digest = "e" * 64
            record = mod.prepared_record(digest, "f" * 64, digest)
            code = """import importlib.util, os, sys
spec=importlib.util.spec_from_file_location('n1',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.fsync=lambda fd: os._exit(77)
m.consume_once(m.prepared_record('e'*64,'f'*64,'e'*64),sys.argv[2],m.path_chain(sys.argv[2]))
"""
            run = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(directory)],
                                 capture_output=True)
            self.assertEqual(run.returncode, 77)
            self.assertEqual(len(list(directory.iterdir())), 1)
            with self.assertRaisesRegex(mod.Hold, "marker-spent"):
                mod.consume_once(record, directory, mod.path_chain(directory))

    def test_two_processes_one_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "protected"
            directory.mkdir(mode=0o700)
            start = Path(temp) / "start"
            code = """import importlib.util, pathlib, sys, time
spec=importlib.util.spec_from_file_location('n1',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
while not pathlib.Path(sys.argv[3]).exists(): time.sleep(.001)
r=m.prepared_record('a'*64,'b'*64,'a'*64)
try: m.consume_once(r,sys.argv[2],m.path_chain(sys.argv[2]))
except m.Hold as e: sys.exit(73 if str(e)=='marker-spent' else 74)
sys.exit(0)
"""
            commands = [sys.executable, "-c", code, str(SCRIPT), str(directory), str(start)]
            children = [subprocess.Popen(commands, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE) for _ in range(2)]
            start.write_bytes(b"go")
            codes = []
            for child in children:
                child.communicate(timeout=10)
                codes.append(child.returncode)
            self.assertEqual(sorted(codes), [0, 73])

    @staticmethod
    def _consume(record, directory, chain):
        try:
            mod.consume_once(record, directory, chain)
        except mod.Hold as exc:
            if str(exc) != "marker-spent":
                raise
            return "spent"
        return "success"


if __name__ == "__main__":
    unittest.main()
