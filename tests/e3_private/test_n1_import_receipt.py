"""Synthetic byte-integrity and fail-closed checks; no compiler or target run."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/verify-e3-n1-import-receipt.py"
SPEC = importlib.util.spec_from_file_location("n1_receipt", SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {}

        def file(name):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            record = {"path": str(path), "sha256": VERIFIER.hashlib.sha256(path.read_bytes()).hexdigest()}
            self.files[name] = path
            return record

        calls = []
        for i in range(39):
            symbol = "bcmp" if i < 14 else "puts"
            obj = "package-native-5.o" if i < 2 else (
                "package-native-6.o" if i < 14 else "private.o")
            calls.append({
                "object": obj, "symbol": symbol, "section": ".text",
                "offset": f"0x{i + 1:x}", "relocation": "R_X86_64_PLT32",
                "addend": -4,
                "producer": {"source": "fixture.c:1", "final_mapping": "fixture.ll:1",
                             "transformation": "fixture lowering"},
                "abi": {"arguments": ["pointer"], "return": "int",
                        "calling_convention": "C", "widths": "64-bit",
                        "pointer_semantics": "borrowed", "address_spaces": "zero",
                        "visibility": "global", "version": "GLIBC_2.2.5"},
                "declaration": "fixture.h:1", "provider": "libc",
            })
        self.receipt = {
            "schema": "e3-n1-import-receipt-v1",
            "status": "candidate-for-independent-review",
            "pins": {
                "source_head": "a" * 40, "source_tree": "b" * 40,
                "image_id": "sha256:" + "c" * 64,
                "compiler_commit": "d" * 40, "builder_sha256": "e" * 64,
                "target_triple": "x86_64-unknown-linux-gnu",
            },
            "objects": {name: file(name) for name in sorted(VERIFIER.OBJECTS)},
            "evidence": {name: file(name + ".txt") for name in sorted(VERIFIER.EVIDENCE)},
            "calls": calls,
            "depfiles": [{"name": name, "file": file(name)} for name in
                         sorted(VERIFIER.DEPFILES)],
            "headers": [{"name": str(self.root / "header.h"), "file": file("header.h")}],
            "provider_files": [{"name": "libc", "file": file("libc.so")}],
            "provider": {
                symbol: {"file": "libc", "symbol_type": "FUNC",
                         "symbol_version": "GLIBC_2.2.5", "soname": "libc.so.6",
                         "binding": "GLOBAL", "visibility": "DEFAULT"}
                for symbol in ("bcmp", "puts")
            } | {
                "symlink_chain": [{"path": str(self.root / "libc.so.6"),
                                   "target": "libc.so"}],
                "loader_path": [str(self.root)],
                "transitive_closure": ["libc"],
            },
        }
        (self.root / "libc.so.6").symlink_to("libc.so")
        trace = self.files["full_header_trace.txt"]
        trace.write_text(str(self.root / "header.h") + "\n")
        self.receipt["evidence"]["full_header_trace"]["sha256"] = (
            VERIFIER.hashlib.sha256(trace.read_bytes()).hexdigest())
        inventories = {
            "normal_imports": ["eshkol_ad_seed_flag", "eshkol_runtime_fatal",
                               "eshkol_taylor_lift_ad_node", "memcmp"],
            "noninstrumentation_imports": ["bcmp", "puts"],
            "sanitizer_imports": ["__asan_init"],
        }
        for name, lines in inventories.items():
            path = self.files[name + ".txt"]
            path.write_text("\n".join(lines) + "\n")
            self.receipt["evidence"][name]["sha256"] = (
                VERIFIER.hashlib.sha256(path.read_bytes()).hexdigest())
        depfile_anchor = {
            "dependencies": {"depfile_sha256": {
                record["name"]: record["file"]["sha256"]
                for record in self.receipt["depfiles"]
            }}
        }
        anchor_file = self.files["depfile_anchor.txt"]
        anchor_file.write_text(json.dumps(depfile_anchor, sort_keys=True))
        self.receipt["evidence"]["depfile_anchor"]["sha256"] = (
            VERIFIER.hashlib.sha256(anchor_file.read_bytes()).hexdigest())
        claims_file = self.files["reviewed_claims.txt"]
        claims_file.write_text(json.dumps(VERIFIER.reviewed_claims(self.receipt),
                                         sort_keys=True))
        self.receipt["evidence"]["reviewed_claims"]["sha256"] = (
            VERIFIER.hashlib.sha256(claims_file.read_bytes()).hexdigest())
        self.expected_pins = copy.deepcopy(self.receipt["pins"])
        self.expected_objects = {name: record["sha256"] for name, record in
                                 self.receipt["objects"].items()}
        self.expected_calls = VERIFIER.call_roster_hash(self.receipt["calls"])
        self.expected_imports = {name: self.receipt["evidence"][name]["sha256"]
                                 for name in VERIFIER.F_IMPORTS}
        self.expected_depfile_anchor = self.receipt["evidence"]["depfile_anchor"]["sha256"]
        self.expected_depfile_paths = {
            record["name"]: record["file"]["path"] for record in self.receipt["depfiles"]
        }
        self.expected_claims = self.receipt["evidence"]["reviewed_claims"]["sha256"]

    def check(self, candidate):
        return VERIFIER.check_receipt(
            candidate, expected_pins=self.expected_pins,
            expected_objects=self.expected_objects,
            expected_calls=self.expected_calls, expected_imports=self.expected_imports,
            expected_depfile_anchor=self.expected_depfile_anchor,
            expected_depfile_paths=self.expected_depfile_paths,
            expected_claims=self.expected_claims)

    def rejects(self, change, phrase):
        candidate = copy.deepcopy(self.receipt)
        change(candidate)
        with self.assertRaisesRegex(VERIFIER.ReceiptError, phrase):
            self.check(candidate)

    def test_complete_synthetic_structure(self):
        self.assertIn("independent", self.check(self.receipt))
        with self.assertRaisesRegex(
                VERIFIER.ReceiptError, "no independently accepted.*claims anchor"):
            VERIFIER.check_receipt(
                self.receipt, expected_pins=self.expected_pins,
                expected_objects=self.expected_objects,
                expected_calls=self.expected_calls,
                expected_imports=self.expected_imports,
                expected_depfile_anchor=self.expected_depfile_anchor,
                expected_depfile_paths=self.expected_depfile_paths)

    def test_missing_producer_or_abi(self):
        self.rejects(lambda x: x["calls"][14]["producer"].update(final_mapping=""),
                     "producer final_mapping")
        self.rejects(lambda x: x["calls"][14]["abi"].update(arguments=[]),
                     "ABI arguments")
        self.rejects(lambda x: x["calls"][14]["abi"].update(calling_convention=""),
                     "calling_convention")
        self.rejects(lambda x: x["calls"][14]["abi"].update(calling_convention="stdcall"),
                     "claims differ")
        self.rejects(lambda x: x["calls"][14].update(provider="other-library"),
                     "call provider differs")
        self.rejects(lambda x: x["calls"][14].update(declaration="other.h:9"),
                     "claims differ")

    def test_relocation_and_inventory_drift(self):
        self.rejects(lambda x: x["calls"][0].update(offset=x["calls"][1]["offset"]),
                     "duplicate.*relocation")
        self.rejects(lambda x: x["calls"][0].update(symbol="puts"), "object/symbol")
        self.rejects(lambda x: x["calls"][0].update(relocation="R_X86_64_PC32"),
                     "changed.*relocation")
        self.rejects(lambda x: x["calls"].pop(), "39 call rows")
        self.rejects(lambda x: x["calls"].append(copy.deepcopy(x["calls"][0])),
                     "39 call rows")
        self.files["noninstrumentation_imports.txt"].write_text("bcmp\nputs\nstrcpy\n")
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "changed noninstrumentation_imports bytes"):
            self.check(self.receipt)

    def test_provider_and_link_drift(self):
        self.rejects(lambda x: x["provider"]["puts"].update(symbol_version=""),
                     "symbol_version")
        self.rejects(lambda x: x["provider"]["puts"].update(symbol_version="GLIBC_99.0"),
                     "call ABI version differs")
        self.rejects(lambda x: x["provider"]["puts"].update(symbol_type="OBJECT"),
                     "claims differ")
        self.rejects(lambda x: x["provider"]["puts"].update(binding="WEAK"),
                     "claims differ")
        altered = copy.deepcopy(self.receipt)
        altered["provider"]["puts"]["symbol_version"] = "GLIBC_99.0"
        for call in altered["calls"]:
            if call["symbol"] == "puts":
                call["abi"]["version"] = "GLIBC_99.0"
        claims_file = self.files["reviewed_claims.txt"]
        claims_file.write_text(json.dumps(VERIFIER.reviewed_claims(altered),
                                         sort_keys=True))
        altered["evidence"]["reviewed_claims"]["sha256"] = (
            VERIFIER.hashlib.sha256(claims_file.read_bytes()).hexdigest())
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "reviewed claims anchor changed"):
            self.check(altered)
        claims_file.write_text(json.dumps(VERIFIER.reviewed_claims(self.receipt),
                                         sort_keys=True))
        self.rejects(lambda x: x["provider"]["puts"].update(file="other"),
                     "unlisted.*provider")
        self.rejects(lambda x: x["provider"].update(loader_path=[]), "loader_path")
        self.rejects(lambda x: x["provider"].update(loader_path=["/tmp/other"]),
                     "claims differ")
        self.rejects(lambda x: x["provider"]["symlink_chain"][0].update(target="other"),
                     "changed provider symlink")
        self.rejects(lambda x: x["evidence"].pop("resolved_link_map"),
                     "evidence records")
        self.files["libc.so"].write_bytes(b"swapped")
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "changed provider_files libc"):
            self.check(self.receipt)

    def test_header_depfile_object_and_image_drift(self):
        self.rejects(lambda x: x["headers"].clear(), "missing headers")
        self.rejects(lambda x: x["depfiles"].pop(), "missing or extra F depfiles")
        substitute = self.root / "private-copy.d"
        substitute.write_bytes(self.files["private.d"].read_bytes())
        self.rejects(
            lambda x: next(row for row in x["depfiles"]
                           if row["name"] == "private.d")["file"].update(path=str(substitute)),
            "path differs from sealed F depfile")
        changed = copy.deepcopy(self.receipt)
        depfile = next(row for row in changed["depfiles"] if row["name"] == "private.d")
        self.files["private.d"].write_bytes(b"changed direct dependency")
        depfile["file"]["sha256"] = VERIFIER.hashlib.sha256(
            self.files["private.d"].read_bytes()).hexdigest()
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "differs from sealed F depfile"):
            self.check(changed)
        self.files["private.d"].write_bytes(b"private.d")
        self.rejects(lambda x: x["headers"][0].update(name="/missing.h"),
                     "missing or extra traced header")
        self.rejects(lambda x: x["pins"].update(image_id="sha256:" + "0" * 63),
                     "image ID")
        self.files["private.o"].write_bytes(b"drift")
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "changed private.o bytes"):
            self.check(self.receipt)

    def test_symlink_and_duplicate_json_key(self):
        original = self.files["libc.so"]
        original.unlink()
        original.symlink_to(self.files["header.h"])
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "unresolved.*symlink"):
            self.check(self.receipt)
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "duplicate JSON key"):
            VERIFIER.json.loads('{"a":1,"a":2}', object_pairs_hook=VERIFIER.pairs_unique)


if __name__ == "__main__":
    unittest.main()
