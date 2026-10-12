"""Only tiny synthetic OCI layouts; no local image or Docker access."""

import copy
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e3_n1_oci_evidence.py"
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
TAR = "application/vnd.oci.image.layer.v1.tar"


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def tar_bytes(name, content):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


class OciEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.layout = self.home / "layout"
        (self.layout / "blobs/sha256").mkdir(parents=True)
        (self.layout / "oci-layout").write_bytes(raw({"imageLayoutVersion": "1.0.0"}))
        self.platform = {"os": "linux", "architecture": "amd64"}
        first = tar_bytes("a", b"first")
        second = tar_bytes("b", b"second")
        self.diff_ids = [digest(first), digest(second)]
        self.layers = [self.blob(gzip.compress(first, mtime=0), TAR + "+gzip"),
                       self.blob(second, TAR)]
        self.config = {"os": "linux", "architecture": "amd64",
                       "rootfs": {"type": "layers", "diff_ids": self.diff_ids}}
        self.config_desc = self.blob(raw(self.config), CONFIG)
        self.manifest = {"schemaVersion": 2, "mediaType": MANIFEST,
                         "config": self.config_desc, "layers": self.layers}
        self.manifest_desc = self.blob(raw(self.manifest), MANIFEST)
        self.manifest_desc["platform"] = self.platform
        self.index = {"schemaVersion": 2, "mediaType": INDEX,
                      "manifests": [self.manifest_desc]}
        self.index_bytes = raw(self.index)
        (self.layout / "index.json").write_bytes(self.index_bytes)
        observations = {name: {"state": "UNMEASURED/HOLD", "records": []}
                        for name in ("os", "tools", "packages", "headers",
                                     "paths", "environment")}
        self.plan = {"schema": "e3-n1-oci-input-v1", "layout": str(self.layout),
                     "index_digest": digest(self.index_bytes),
                     "platform": self.platform,
                     "limits": {"max_blob_bytes": 100000,
                                "max_expanded_layer_bytes": 100000,
                                "max_total_expanded_bytes": 200000,
                                "max_evidence_bytes": 10000,
                                "max_seconds": 10, "max_rss_bytes": 1024**3},
                     "observations": observations}

    def blob(self, data, media_type):
        name = hashlib.sha256(data).hexdigest()
        (self.layout / "blobs/sha256" / name).write_bytes(data)
        return {"mediaType": media_type, "digest": "sha256:" + name,
                "size": len(data)}

    def rebuild(self):
        self.config_desc = self.blob(raw(self.config), CONFIG)
        self.manifest["config"] = self.config_desc
        self.manifest_desc = self.blob(raw(self.manifest), MANIFEST)
        self.manifest_desc["platform"] = self.platform
        self.index["manifests"] = [self.manifest_desc]
        self.index_bytes = raw(self.index)
        (self.layout / "index.json").write_bytes(self.index_bytes)
        self.plan["index_digest"] = digest(self.index_bytes)

    def run_cli(self, plan=None, input_bytes=None):
        path = self.home / "input.json"
        path.write_bytes(input_bytes if input_bytes is not None else raw(plan or self.plan))
        return subprocess.run([sys.executable, str(SCRIPT), str(path)],
                              capture_output=True, check=False)

    def assert_hold(self, plan=None, input_bytes=None):
        result = self.run_cli(plan, input_bytes)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"HOLD:", result.stderr)

    def test_exact_chain_checksum_and_unmeasured_rows(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(result.stdout, raw(evidence))
        checksum = evidence.pop("checksum_sha256")
        self.assertEqual(checksum, digest(raw(evidence)))
        expected_chain = digest((self.diff_ids[0] + " " + self.diff_ids[1]).encode())
        self.assertEqual(evidence["oci"]["rootfs"]["chain_id"], expected_chain)
        self.assertEqual(evidence["oci"]["rootfs"]["diff_ids"], self.diff_ids)
        self.assertEqual(evidence["status"], "HOLD")
        self.assertEqual(evidence["oci"]["index_digest"], self.plan["index_digest"])

    def test_external_record_is_byte_bound_but_unverified(self):
        source = self.home / "observation.txt"
        source.write_bytes(b"synthetic host claim\n")
        self.plan["observations"]["tools"] = {
            "state": "external-unverified",
            "records": [{"subject": "compiler", "value": "/synthetic/clang",
                         "evidence": {"path": str(source), "size": source.stat().st_size,
                                      "sha256": digest(source.read_bytes()),
                                      "producer": "synthetic-test", "method": "fixture"}}]}
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["observations"]["tools"]["state"],
                         "external-unverified")
        source.write_bytes(b"changed\n")
        self.assert_hold()

    def test_byte_and_shape_mutants_hold(self):
        cases = []
        changed = copy.deepcopy(self.plan)
        changed["index_digest"] = "sha256:" + "0" * 64
        cases.append(changed)
        changed = copy.deepcopy(self.plan)
        changed["limits"]["max_blob_bytes"] = True
        cases.append(changed)
        changed = copy.deepcopy(self.plan)
        del changed["observations"]["headers"]
        cases.append(changed)
        changed = copy.deepcopy(self.plan)
        changed["observations"]["paths"]["state"] = "external-unverified"
        cases.append(changed)
        for case in cases:
            with self.subTest(case=case):
                self.assert_hold(case)
        self.assert_hold(input_bytes=json.dumps(self.plan).encode())
        self.assert_hold(input_bytes=raw(self.plan).replace(b'"schema":', b'"schema":"duplicate","schema":', 1))

    def test_chain_schema_and_platform_mutants_hold(self):
        self.index["schemaVersion"] = True
        self.rebuild()
        self.assert_hold()
        self.index["schemaVersion"] = 2
        self.rebuild()
        self.index["manifests"].append(copy.deepcopy(self.manifest_desc))
        self.index_bytes = raw(self.index)
        (self.layout / "index.json").write_bytes(self.index_bytes)
        self.plan["index_digest"] = digest(self.index_bytes)
        self.assert_hold()
        self.index["manifests"].pop()
        self.config["rootfs"]["type"] = "wrong"
        self.rebuild()
        self.assert_hold()
        self.config["rootfs"]["type"] = "layers"
        self.config["rootfs"]["diff_ids"][0] = "sha256:" + "0" * 64
        self.rebuild()
        self.assert_hold()

    def test_layer_mutants_hold(self):
        blob = self.layout / "blobs/sha256" / self.layers[0]["digest"][7:]
        original = blob.read_bytes()
        blob.write_bytes(original + b"trailing")
        self.assert_hold()
        blob.write_bytes(original)
        blob.write_bytes(original[:-1])
        self.assert_hold()
        blob.write_bytes(original)
        self.plan["limits"]["max_expanded_layer_bytes"] = 100
        self.assert_hold()
        self.plan["limits"]["max_expanded_layer_bytes"] = 100000
        blob.unlink()
        blob.symlink_to(self.layout / "blobs/sha256" / self.layers[1]["digest"][7:])
        self.assert_hold()

    def test_complete_second_gzip_member_is_rejected(self):
        first = self.layout / "blobs/sha256" / self.layers[0]["digest"][7:]
        joined = first.read_bytes() + gzip.compress(b"second member", mtime=0)
        self.manifest["layers"][0] = self.blob(joined, TAR + "+gzip")
        self.rebuild()
        self.assert_hold()

    def test_total_cap_preempts_later_gzip_error(self):
        # A corrupt tail would fail later; the total cap must stop the first
        # layer before the decoder reaches it.
        large = tar_bytes("large", b"x" * 150000)
        damaged = gzip.compress(large, compresslevel=0, mtime=0)[:-4]
        self.manifest["layers"][0] = self.blob(damaged, TAR + "+gzip")
        self.plan["limits"]["max_blob_bytes"] = 300000
        self.plan["limits"]["max_expanded_layer_bytes"] = 300000
        self.plan["limits"]["max_total_expanded_bytes"] = 4096
        self.rebuild()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"total-expanded-cap", result.stderr)

    def test_non_utf8_oci_json_holds(self):
        (self.layout / "oci-layout").write_bytes(
            json.dumps({"imageLayoutVersion": "1.0.0"}).encode("utf-16"))
        self.assert_hold()
        self.index_bytes = json.dumps(self.index).encode("utf-16")
        (self.layout / "index.json").write_bytes(self.index_bytes)
        self.plan["index_digest"] = digest(self.index_bytes)
        (self.layout / "oci-layout").write_bytes(raw({"imageLayoutVersion": "1.0.0"}))
        self.assert_hold()

    def test_unpaired_surrogate_and_numeric_overflow_hold(self):
        self.manifest_desc["annotations"] = {"synthetic": "\ud800"}
        self.index_bytes = raw(self.index)
        (self.layout / "index.json").write_bytes(self.index_bytes)
        self.plan["index_digest"] = digest(self.index_bytes)
        self.assert_hold()
        self.assert_hold(input_bytes=b'{"schema":1e999}\n')


if __name__ == "__main__":
    unittest.main()
