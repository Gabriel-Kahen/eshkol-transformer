"""Focused schema, failure, and real native A2 benchmark witnesses."""
import copy
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if __package__:
    from . import attention_benchmark as attention
else:
    import attention_benchmark as attention


class AttentionFormatTests(unittest.TestCase):
    def test_checked_in_definition_and_smoke_isolation(self):
        definition = attention.load_definition(attention.ROOT/"benchmarks/attention_v1.json")
        self.assertEqual(definition, attention.CONTRACT)
        self.assertEqual(attention.encode_definition(definition),
                         (attention.ROOT/"benchmarks/attention_v1.json").read_bytes())
        with self.assertRaisesRegex(attention.codec.BenchmarkFormatError, "unsupported benchmark format"):
            attention.codec.load_definition(attention.encode_definition(definition))

    def test_unsupported_contracts_and_invalid_counts_reject(self):
        for name, value in [("device","cuda:0"), ("dtype","f64"), ("backend","fake"),
                            ("shapes",[[1,2,2,1024,1024,64]]), ("deterministic",1),
                            ("operations",["rope"]), ("warmup",-1), ("iterations",0),
                            ("iterations",True), ("repetitions",101), ("timeout_ms",0)]:
            with self.subTest(name=name,value=value):
                definition = copy.deepcopy(attention.CONTRACT)
                definition[name] = value
                with self.assertRaises(attention.codec.BenchmarkFormatError):
                    attention.encode_definition(definition)

    def test_corrupt_noncanonical_duplicate_unknown_and_version_reject(self):
        encoded = attention.encode_definition(attention.CONTRACT)
        cases = [encoded.replace(b'"warmup":10',b'"warmup":11'), b" " + encoded,
                 encoded.replace(b'"version":1',b'"version":2'),
                 encoded.replace(b'"version":1',b'"version":1,"version":1')]
        for data in cases:
            with self.subTest(data=data[-50:]):
                with self.assertRaises(attention.codec.BenchmarkFormatError):
                    attention.load_definition(data)
        malformed = dict(attention.CONTRACT, unknown=1)
        with self.assertRaises(attention.codec.BenchmarkFormatError):
            attention.encode_definition(malformed)

    def test_native_protocol_rejects_nonfinite_wrong_shape_and_provider(self):
        valid = {"version":1,"provider":"a2.cpu-f32.serial","dispatch_elapsed_ns":1,
                 "output_bits":"3f80000040000000"}
        for name, value in [("output_bits","7f80000040000000"), ("output_bits","0000000000000000"),
                            ("output_bits","3f800000"), ("output_bits","z"*16),
                            ("provider","fallback"), ("dispatch_elapsed_ns",0), ("version",True)]:
            with self.subTest(name=name,value=value):
                changed = dict(valid, **{name:value})
                with self.assertRaises(attention.codec.BenchmarkFormatError):
                    attention.native_output(json.dumps(changed).encode(), attention.SHAPES[0], "forward")


@unittest.skipUnless(platform.system() == "Linux" and shutil.which("clang"), "Linux/clang required")
class NativeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.target, cls.helper, cls.compiler = attention.build("clang", cls.directory)
        cls.definition = copy.deepcopy(attention.CONTRACT)
        cls.definition.update(warmup=1,iterations=3,repetitions=2)
        cls.definition_path = cls.directory/"definition.json"
        cls.definition_path.write_bytes(attention.encode_definition(cls.definition))
        cls.report_path = cls.directory/"report.json"
        with mock.patch.object(attention,"build",return_value=(cls.target,cls.helper,cls.compiler)):
            attention.run(cls.definition_path,cls.report_path,"clang",cls.directory)
        cls.report = attention.load_report(cls.report_path)

    def invoke(self, *args):
        return subprocess.run([str(self.target),*args],capture_output=True,check=False)

    def synthetic_v2_report(self):
        """A format fixture, not evidence of a supported measurement."""
        report = copy.deepcopy(self.report)
        lock_raw = (attention.ROOT/"toolchain/eshkol.lock").read_text()
        lock = attention.parse_support_tsv(lock_raw, "fixture lock")
        report["stable"]["project"]["dirty"] = False
        report["stable"]["compiler"]["path"] = "/usr/bin/clang-21"
        report["stable"]["compiler"]["version"] = "clang version 21.1.8"
        provenance_raw = (f"eshkol_repository\t{lock['eshkol_repository']}\n"
                          f"eshkol_commit\t{lock['eshkol_commit']}\n"
                          f"eshkol_version\t{lock['eshkol_version']}\n"
                          f"llvm_version\t{lock['llvm_version']}\n"
                          f"cc_path\t/usr/bin/clang-21\n"
                          f"cc_version\t{lock['clang_version']}\n"
                          f"cxx_version\t{lock['clang_version']}\n"
                          f"eshkol_binary_sha256\t{'0'*64}\n")
        report["stable"]["sources"].update({
            p: attention.common.sha256_file(attention.ROOT/p) for p in attention.SUPPORT_SOURCES})
        report["stable"]["support_evidence"] = {
            "lock_tsv": lock_raw, "provenance_tsv": provenance_raw,
            "provenance_sha256": hashlib.sha256(provenance_raw.encode()).hexdigest()}
        report["volatile"]["support_status"] = "supported"
        report["volatile"]["environment"]["architecture"] = "x86_64"
        report["volatile"]["support_evidence"] = {
            "os_id": "ubuntu", "os_version_id": "22.04", "architecture": "x86_64"}
        return report

    def test_real_dispatch_all_rows_and_repeated_outputs(self):
        cases = self.report["volatile"]["cases"]
        self.assertEqual(len(cases),20)
        for case in cases:
            self.assertEqual(len(case["samples"]),2)
            for sample in case["samples"]:
                self.assertGreater(sample["dispatch_elapsed_ns"],0)
                self.assertGreaterEqual(sample["process_elapsed_ns"],sample["dispatch_elapsed_ns"])
                self.assertGreater(sample["process_peak_rss_kib"],0)
                self.assertEqual(sample["output_sha256"],case["output_sha256"])
        self.assertEqual(self.report["volatile"]["support_status"],"compatibility-only")
        self.assertEqual(set(self.report["stable"]["sources"]),set(attention.SOURCES))

    def test_t1_known_forward_and_backward_outputs(self):
        for operation, expected in [("forward",[-5/16,2/16]),
                                    ("backward",[0,0,0,0,-2/16,5/16])]:
            result = self.invoke("0",operation,"0","1")
            self.assertEqual(result.returncode,0,result.stderr)
            value = json.loads(result.stdout)
            bits = value["output_bits"]
            actual = [struct.unpack("!f",bytes.fromhex(bits[i:i+8]))[0] for i in range(0,len(bits),8)]
            self.assertEqual(actual,expected)
            elapsed,digest = attention.native_output(result.stdout,attention.SHAPES[0],operation)
            self.assertGreater(elapsed,0)
            self.assertEqual(digest,hashlib.sha256(bytes.fromhex(bits)).hexdigest())

    def test_native_rejects_invalid_cli_without_stdout(self):
        for args in [[],["10","forward","1","1"],["0","rope","1","1"],
                     ["0","forward","1","0"],["0","forward","-1","1"],
                     ["0","forward","1","1000001"],["0junk","forward","1","1"],
                     ["9999999999999999999999999999999","forward","1","1"]]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,b"")
                self.assertIn(b"attention benchmark:",result.stderr)

    def test_report_tampering_and_unsupported_claims_reject(self):
        mutations = [lambda r:r["stable"].update(definition_sha256="0"*64),
                     lambda r:r["stable"]["sources"].update(unknown="0"*64),
                     lambda r:r["volatile"].update(support_status="supported"),
                     lambda r:r["volatile"]["environment"].update(logical_cpus=True),
                     lambda r:r["volatile"]["cases"].pop(),
                     lambda r:r["volatile"]["cases"][0]["samples"][0].update(output_sha256="0"*64),
                     lambda r:r["volatile"]["cases"][0]["samples"][0].update(dispatch_elapsed_ns=True),
                     lambda r:r["volatile"]["cases"][0]["samples"][0].update(process_elapsed_ns=1),
                     lambda r:r["volatile"]["cases"][0]["summary"].update(dispatch_ns_min=1)]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(mutate=mutate):
                with self.assertRaises(attention.codec.BenchmarkFormatError):
                    attention.encode_report(report)
        encoded = attention.encode_report(self.report)
        with self.assertRaisesRegex(attention.codec.BenchmarkFormatError,"checksum mismatch"):
            attention.load_report(encoded.replace(b'"compatibility-only"',b'"supported"'))

    def test_v2_schema_and_wrong_or_forged_evidence(self):
        report = self.synthetic_v2_report()
        encoded = attention.encode_report(report, version=2)
        self.assertEqual(attention.load_report(encoded), report)
        with self.assertRaises(attention.codec.BenchmarkFormatError):
            attention.encode_report(report)
        mutations = [
            lambda r: r["stable"]["project"].update(dirty=True),
            lambda r: r["volatile"]["support_evidence"].update(os_id="cachyos"),
            lambda r: r["volatile"]["support_evidence"].update(architecture="aarch64"),
            lambda r: r["stable"]["sources"].update({"toolchain/eshkol.lock": "0"*64}),
            lambda r: r["stable"]["support_evidence"].update(provenance_sha256="0"*64),
            lambda r: r["stable"]["compiler"].update(path="/usr/bin/gcc"),
            lambda r: r["stable"]["compiler"].update(version="clang version 20.0.0"),
            lambda r: r["volatile"].update(support_status="compatibility-only"),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.subTest(mutate=mutate):
                with self.assertRaises(attention.codec.BenchmarkFormatError):
                    attention.encode_report(changed, version=2)
        changed = copy.deepcopy(report)
        evidence = changed["stable"]["support_evidence"]
        evidence["provenance_tsv"] = evidence["provenance_tsv"].replace(
            "cc_version\t21.1.8", "cc_version\t20.0.0")
        evidence["provenance_sha256"] = hashlib.sha256(
            evidence["provenance_tsv"].encode()).hexdigest()
        with self.assertRaisesRegex(attention.codec.BenchmarkFormatError, "toolchain cc_version mismatch"):
            attention.encode_report(changed, version=2)

    def test_supported_gate_failure_preserves_previous_report(self):
        path = self.directory/"supported-preserved.json"
        path.write_bytes(b"previous")
        with mock.patch.dict(os.environ, {"ESHKOL_ALLOW_UNSUPPORTED_HOST": "1"}):
            with mock.patch.object(attention.subprocess, "run") as process:
                with self.assertRaisesRegex(attention.codec.BenchmarkFormatError, "host override"):
                    attention.run(self.definition_path, path, "clang-21", self.directory,
                                  supported=True)
                process.assert_not_called()
        self.assertEqual(path.read_bytes(), b"previous")
        with mock.patch.object(attention, "supported_probe", side_effect=attention.codec.BenchmarkFormatError("wrong toolchain")):
            with self.assertRaisesRegex(attention.codec.BenchmarkFormatError, "wrong toolchain"):
                attention.run(self.definition_path, path, "clang-21", self.directory,
                              supported=True)
        self.assertEqual(path.read_bytes(), b"previous")

    def test_measurement_failure_and_nondeterminism_preserve_report(self):
        path = self.directory/"preserved.json"
        for failure in [attention.codec.BenchmarkFormatError("timed out"), None]:
            path.write_bytes(b"previous")
            sample = {"dispatch_elapsed_ns":1,"process_elapsed_ns":2,"process_peak_rss_kib":3}
            with mock.patch.object(attention,"build",return_value=(self.target,self.helper,self.compiler)):
                with mock.patch.object(attention,"measure",side_effect=failure or [(sample,"a"*64),(sample,"b"*64)]):
                    with self.assertRaises(attention.codec.BenchmarkFormatError):
                        attention.run(self.definition_path,path,"clang",self.directory)
            self.assertEqual(path.read_bytes(),b"previous")


if __name__ == "__main__":
    unittest.main()
