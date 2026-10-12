"""Synthetic byte-integrity and fail-closed checks; no compiler or target run."""

import copy
import importlib.util
import json
from pathlib import Path
import stat
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
        for directory in ("image/include", "source", "build"):
            (self.root / directory).mkdir(parents=True)

        def file(name):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            record = {"path": str(path), "sha256": VERIFIER.hashlib.sha256(path.read_bytes()).hexdigest(),
                      "mode": stat.S_IMODE(path.stat().st_mode), "size": path.stat().st_size}
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
                "candidate_kind": "e3-numerical-f",
            },
            "objects": {name: file(name) for name in sorted(VERIFIER.OBJECTS)},
            "evidence": {name: file(name + ".txt") for name in sorted(VERIFIER.EVIDENCE)},
            "calls": calls,
            "depfiles": [{"name": name, "file": file(name)} for name in
                         sorted(VERIFIER.DEPFILES)],
            "headers": [{"name": str(self.root / "image/include/header.h"),
                         "file": file("image/include/header.h")}],
            "provider_files": [{"name": "libc", "file": file("image/libc.so")}],
            "provider": {
                symbol: {"file": "libc", "symbol_type": "FUNC",
                         "symbol_version": "GLIBC_2.2.5", "soname": "libc.so.6",
                         "binding": "GLOBAL", "visibility": "DEFAULT"}
                for symbol in ("bcmp", "puts")
            } | {
                "symlink_chain": [{"path": str(self.root / "image/libc.so.6"),
                                   "target": "libc.so"}],
                "loader_path": [str(self.root / "image")],
                "transitive_closure": ["libc"],
            },
        }
        (self.root / "image/libc.so.6").symlink_to("libc.so")
        source = file("source/fixture.c")
        header = str(self.root / "image/include/header.h")
        self.receipt["dependencies"] = {
            "roots": {name: str(self.root / name) for name in ("image", "source", "build")},
            "include_roots": [str(self.root / "image/include")],
            "sources": [{"name": source["path"], "origin": "source", "file": source}],
            "header_origins": {header: "image"},
            "depfile_members": {name: sorted([source["path"], header])
                                for name in VERIFIER.DEPFILES},
            "depfile_path_map": {source["path"]: source["path"], header: header},
            "include_edges": [{"from": source["path"], "to": header}],
        }
        for name in VERIFIER.DEPFILES:
            path = self.files[name]
            path.write_text(f"/build/{name[:-2]}.o: {source['path']} {header}\n")
            record = next(row["file"] for row in self.receipt["depfiles"]
                          if row["name"] == name)
            record["sha256"] = VERIFIER.hashlib.sha256(path.read_bytes()).hexdigest()
            record["size"] = path.stat().st_size
        self.receipt["build_objects"] = [
            {"variant": variant, "name": "private.o", "file": file(f"build/{variant}.o")}
            for variant in ("normal", "sanitized")
        ]
        inventory = self.files["object_inventory.txt"]
        inventory.write_text(json.dumps(self.receipt["build_objects"], sort_keys=True))
        self.receipt["evidence"]["object_inventory"]["sha256"] = (
            VERIFIER.hashlib.sha256(inventory.read_bytes()).hexdigest())
        self.receipt["evidence"]["object_inventory"]["size"] = inventory.stat().st_size
        trace = self.files["full_header_trace.txt"]
        trace.write_text(header + "\n")
        self.receipt["evidence"]["full_header_trace"]["sha256"] = (
            VERIFIER.hashlib.sha256(trace.read_bytes()).hexdigest())
        self.receipt["evidence"]["full_header_trace"]["size"] = trace.stat().st_size
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
            self.receipt["evidence"][name]["size"] = path.stat().st_size
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
        self.receipt["evidence"]["depfile_anchor"]["size"] = anchor_file.stat().st_size
        claims_file = self.files["reviewed_claims.txt"]
        claims_file.write_text(json.dumps(VERIFIER.reviewed_claims(self.receipt),
                                         sort_keys=True))
        self.receipt["evidence"]["reviewed_claims"]["sha256"] = (
            VERIFIER.hashlib.sha256(claims_file.read_bytes()).hexdigest())
        self.receipt["evidence"]["reviewed_claims"]["size"] = claims_file.stat().st_size
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
        self.expected_build_roster = sorted((item["variant"], item["name"])
                                            for item in self.receipt["build_objects"])

    def check(self, candidate):
        return VERIFIER.check_receipt(
            candidate, expected_pins=self.expected_pins,
            expected_objects=self.expected_objects,
            expected_calls=self.expected_calls, expected_imports=self.expected_imports,
            expected_depfile_anchor=self.expected_depfile_anchor,
            expected_depfile_paths=self.expected_depfile_paths,
            expected_claims=self.expected_claims,
            expected_build_roster=self.expected_build_roster)

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
                expected_depfile_paths=self.expected_depfile_paths,
                expected_build_roster=self.expected_build_roster)

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
                                    "changed noninstrumentation_imports (size|bytes)"):
            self.check(self.receipt)

    def test_same_byte_raw_path_substitution(self):
        alternate = self.root / "argv-copy.txt"
        alternate.write_bytes(self.files["argv.txt"].read_bytes())
        self.rejects(lambda x: x["evidence"]["argv"].update(path=str(alternate)),
                     "claims differ from reviewed anchor")

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
        altered["evidence"]["reviewed_claims"]["size"] = claims_file.stat().st_size
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "reviewed claims anchor changed"):
            self.check(altered)
        claims_file.write_text(json.dumps(VERIFIER.reviewed_claims(self.receipt),
                                         sort_keys=True))
        self.rejects(lambda x: x["provider"]["puts"].update(file="other"),
                     "unlisted.*provider")
        self.rejects(lambda x: x["provider"].update(loader_path=[]), "loader_path")
        self.rejects(lambda x: x["provider"].update(loader_path=["/tmp/other"]),
                     "host or unresolved loader search directory")
        self.rejects(lambda x: x["provider"]["symlink_chain"][0].update(target="other"),
                     "changed provider symlink")
        self.rejects(lambda x: x["evidence"].pop("resolved_link_map"),
                     "evidence records")
        self.files["image/libc.so"].write_bytes(b"swapped")
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
        original = self.files["private.d"].read_bytes()
        self.files["private.d"].write_bytes(b"changed direct dependency")
        depfile["file"]["sha256"] = VERIFIER.hashlib.sha256(
            self.files["private.d"].read_bytes()).hexdigest()
        depfile["file"]["size"] = self.files["private.d"].stat().st_size
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "differs from sealed F depfile"):
            self.check(changed)
        self.files["private.d"].write_bytes(original)
        self.rejects(lambda x: x["headers"][0].update(name="/missing.h"),
                     "missing or extra traced header")
        self.rejects(lambda x: x["pins"].update(image_id="sha256:" + "0" * 63),
                     "image ID")
        self.files["private.o"].write_bytes(b"drift")
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "changed private.o (size|bytes)"):
            self.check(self.receipt)

    def test_dependency_closure_and_host_drift(self):
        header = str(self.root / "image/include/header.h")
        self.rejects(lambda x: x["dependencies"]["include_edges"].clear(),
                     "missing direct/transitive includes")
        self.rejects(lambda x: x["dependencies"]["header_origins"].update({header: "source"}),
                     "host or unresolved header")
        self.rejects(lambda x: x["dependencies"]["depfile_members"].pop("private.d"),
                     "missing or extra depfile members")
        self.rejects(lambda x: x["dependencies"]["depfile_members"]["private.d"].remove(header),
                     "private.d members differ from pinned depfile")
        self.rejects(lambda x: x["dependencies"]["depfile_path_map"].pop(header),
                     "members differ from pinned depfile")
        self.rejects(lambda x: x["provider_files"][0]["file"].update(
            path=str(self.root / "source/fixture.c")), "changed provider_files libc")
        self.rejects(lambda x: x["pins"].update(candidate_kind="other-tuple"),
                     "candidate source/image/compiler/flags pins changed")
        self.files["image/include/header.h"].chmod(0o600)
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "changed headers.*mode"):
            self.check(self.receipt)

    def test_empty_sanitizer_inventory(self):
        path = self.files["sanitizer_imports.txt"]
        path.write_text("")
        candidate = copy.deepcopy(self.receipt)
        candidate["evidence"]["sanitizer_imports"]["sha256"] = (
            VERIFIER.hashlib.sha256(path.read_bytes()).hexdigest())
        candidate["evidence"]["sanitizer_imports"]["size"] = 0
        expected = dict(self.expected_imports)
        expected["sanitizer_imports"] = candidate["evidence"]["sanitizer_imports"]["sha256"]
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "invalid sanitizer_imports inventory"):
            VERIFIER.check_receipt(
                candidate, expected_pins=self.expected_pins,
                expected_objects=self.expected_objects,
                expected_calls=self.expected_calls, expected_imports=expected,
                expected_depfile_anchor=self.expected_depfile_anchor,
                expected_depfile_paths=self.expected_depfile_paths,
                expected_claims=self.expected_claims,
                expected_build_roster=self.expected_build_roster)

    def test_normal_sanitized_object_inventory_drift(self):
        self.rejects(lambda x: x["build_objects"].pop(),
                     "missing normal/sanitized object variant")
        self.rejects(lambda x: x["build_objects"][1]["file"].update(
            sha256="0" * 64), "changed build object sanitized/private.o bytes")
        path = self.files["build/sanitized.o"]
        path.chmod(0o600)
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "changed build object sanitized/private.o mode"):
            self.check(self.receipt)

    def test_missing_roster_and_malformed_anchor(self):
        with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                    "no independently accepted.*object roster"):
            VERIFIER.check_receipt(
                self.receipt, expected_pins=self.expected_pins,
                expected_objects=self.expected_objects,
                expected_calls=self.expected_calls,
                expected_imports=self.expected_imports,
                expected_depfile_anchor=self.expected_depfile_anchor,
                expected_depfile_paths=self.expected_depfile_paths,
                expected_claims=self.expected_claims)
        anchor = self.files["depfile_anchor.txt"]
        anchor.write_text('{"dependencies": []}')
        candidate = copy.deepcopy(self.receipt)
        candidate["evidence"]["depfile_anchor"]["sha256"] = (
            VERIFIER.hashlib.sha256(anchor.read_bytes()).hexdigest())
        candidate["evidence"]["depfile_anchor"]["size"] = anchor.stat().st_size
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "invalid depfile anchor"):
            VERIFIER.check_receipt(
                candidate, expected_pins=self.expected_pins,
                expected_objects=self.expected_objects,
                expected_calls=self.expected_calls,
                expected_imports=self.expected_imports,
                expected_depfile_anchor=candidate["evidence"]["depfile_anchor"]["sha256"],
                expected_depfile_paths=self.expected_depfile_paths,
                expected_claims=self.expected_claims,
                expected_build_roster=self.expected_build_roster)

    def test_symlink_and_duplicate_json_key(self):
        original = self.files["image/libc.so"]
        original.unlink()
        original.symlink_to(self.files["image/include/header.h"])
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "unresolved.*symlink"):
            self.check(self.receipt)
        with self.assertRaisesRegex(VERIFIER.ReceiptError, "duplicate JSON key"):
            VERIFIER.json.loads('{"a":1,"a":2}', object_pairs_hook=VERIFIER.pairs_unique)

    def test_provider_symlink_own_location_cannot_escape_image(self):
        for path in (self.root / "image/../source/escape-link",
                     self.root / "image/portal/escape-link"):
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.receipt)
                if "portal" in path.parts:
                    (self.root / "image/portal").symlink_to(self.root / "source")
                actual = self.root / "source/escape-link"
                actual.symlink_to(self.files["image/libc.so"])
                candidate["provider"]["symlink_chain"][0] = {
                    "path": str(path), "target": str(self.files["image/libc.so"])}
                claims = self.files["reviewed_claims.txt"]
                claims.write_text(json.dumps(VERIFIER.reviewed_claims(candidate),
                                             sort_keys=True))
                candidate["evidence"]["reviewed_claims"]["sha256"] = (
                    VERIFIER.hashlib.sha256(claims.read_bytes()).hexdigest())
                candidate["evidence"]["reviewed_claims"]["size"] = claims.stat().st_size
                self.expected_claims = candidate["evidence"]["reviewed_claims"]["sha256"]
                try:
                    with self.assertRaisesRegex(VERIFIER.ReceiptError,
                                                "host provider symlink"):
                        self.check(candidate)
                finally:
                    actual.unlink()


if __name__ == "__main__":
    unittest.main()
