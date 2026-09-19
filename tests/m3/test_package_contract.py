"""Exact accepted M3 facade and source-only packaging policy checks."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tests.m3.check_public_closure import validate

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = (
    ("model-forward", 3), ("model-output-logits", 1),
    ("model-output-loss", 1), ("model-output-rng", 1),
    ("model-output-release!", 1), ("diagnostic-output-vjp!", 4),
    ("diagnostic-model-logits-bits", 1), ("diagnostic-model-logits-release!", 1),
)


def manifest(suffix):
    return (ROOT / f"native/m3_package_{suffix}.txt").read_text().splitlines()


class PackageContract(unittest.TestCase):
    def test_predecessor_pin_inventory(self):
        result = subprocess.run(["sha256sum", "--quiet", "-c",
            str(ROOT / "tests/n3k/expected/predecessor_sources.sha256")],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_surface_and_arity(self):
        facade = (ROOT / "lib/transformer/model.esk").read_text()
        bridge = (ROOT / "native/m3_package_bridge.c").read_text()
        self.assertEqual(re.search(r"\(provide\s+(.*?)\)", facade, re.S)[1].split(),
                         [name for name, _ in OPERATIONS])
        self.assertEqual(re.findall(r"\(require ([^)]+)\)", facade), ["transformer.error_consumer"])
        self.assertEqual(bridge.count('#include "m3t_package_bridge.c"'), 1)
        additions = []
        for name, boxed_arity in OPERATIONS:
            suffix = name.rstrip("!")
            symbol = suffix.replace("-", "_")
            public = f"et_e1b_public_m3_{symbol}_v1"
            private = f"et_e1b_private_m3_{symbol}_cabi_v1"
            params = re.search(r"\(define \(" + re.escape(name) + r"([^)]*)\)", facade)[1].split()
            self.assertEqual(params, ["model", "input-ids", ".", "opts"] if name == "model-forward"
                             else ["first", "second", "third", "fourth"][:boxed_arity])
            ffi = re.search(r"\(extern void m3-boxed-" + re.escape(suffix) +
                            r"\s+((?:ptr\s+)+):real " + public, facade)[1].split()
            self.assertEqual(len(ffi), boxed_arity + 1)
            self.assertEqual(len(re.search(r"void " + public + r"\(([^)]*)\)", bridge)[1].split(",")), boxed_arity + 1)
            self.assertIn(f"m3-public-{name} {private}", manifest("private_renames"))
            additions.append(public)
        self.assertEqual(len(manifest("public_exports")), 87)
        self.assertEqual(len(manifest("defined_symbols")), 93)
        self.assertEqual(manifest("defined_symbols"), manifest("public_strings"))
        predecessor = (ROOT / "native/m3t_package_public_exports.txt").read_text().splitlines()
        self.assertEqual(set(manifest("public_exports")) - set(predecessor), set(additions))
        for suffix in ("public_exports", "defined_symbols", "private_renames", "undefined_symbols"):
            self.assertEqual(manifest(suffix), sorted(set(manifest(suffix))))

    def test_installed_closure_and_single_registry(self):
        self.assertEqual(manifest("facades"), [f"transformer/{name}.esk" for name in (
            "config", "diagnostic_transport", "error_consumer", "error_public", "model", "module", "tokenizer")])
        self.assertEqual(manifest("archive_members"), ["m3_package.o"])
        self.assertEqual(manifest("native_objects"), ["n2/n2_primitives_provider.o", "n3k/n3k_primitives_provider.o", "a2/a2_attention_provider.o"])
        root = (ROOT / "native/m3_package_root.esk").read_text()
        self.assertEqual(re.findall(r'\(load "([^"]+)"\)', root),
                         ["m3t_package_root.esk", "m3_schedule.esk", "m3_model_extension.esk"])
        for name in manifest("facades"):
            for imported in re.findall(r"\(require transformer\.([^\s)]+)\)", (ROOT / "lib" / name).read_text()):
                self.assertIn(f"transformer/{imported}.esk", manifest("facades"))

    def test_native_tuple_replaces_i1_and_transport_exactly_once(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        branch = builder.split('    package_policy=m3-model-aggregate\n', 1)[1].split("\nelif ", 1)[0]
        sources = re.findall(r'"\$\{PROJECT_ROOT\}/([^"\n]+\.c)"', branch)
        self.assertEqual(sources, ["native/data_io.c", "native/checkpoint_io.c",
            "native/kernel_abi.c", "src/eshkol_transformer/m3_i64_integration.c",
            "native/t1_i64_shell.c", "src/eshkol_transformer/m3t_f32_integration.c",
            "src/eshkol_transformer/m3_model.c"])
        self.assertNotIn("native/i64_tensor.c", sources)
        self.assertNotIn("native/f32_tensor.c", sources)
        self.assertNotIn("src/eshkol_transformer/m3t_transport.c", sources)

    def test_public_closure_requires_all_seven_installed_facades(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory)
            source = ROOT / "tests/m3/test_package_contract.py"
            for name in manifest("facades"):
                path = installed / "facades" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("; fixture\n")
            dependencies = [str(source)] + [str(installed / "facades" / name) for name in manifest("facades")]
            depfile = installed / "public.d"
            for invalid in ([], dependencies[:-1], dependencies + dependencies[-1:],
                            dependencies + [str(ROOT / "native/m3_package_root.esk")]):
                depfile.write_text("public.o: " + " ".join(invalid) + "\n")
                with self.assertRaises(ValueError):
                    validate(installed, source, depfile)
            depfile.write_text("public.o: " + " ".join(dependencies) + "\n")
            self.assertEqual(validate(installed, source, depfile), dependencies)

    def test_m3_tuple_rejects_altered_or_aliased_inputs_before_toolchain(self):
        script = '''PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$1/native/m3_package_bridge.c
raw_package_renames=$1/native/m3_package_private_renames.txt
raw_public_exports=$1/native/m3_package_public_exports.txt
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src")
die() { echo "$*" >&2; exit 1; }
source "$1/scripts/m3-package-policy.sh"
'''
        for root in (str(ROOT / "native/../native/m3_package_root.esk"),
                     str(ROOT / "native/m3t_package_root.esk")):
            result = subprocess.run(["/usr/bin/bash", "-eu", "-c", script, "policy-test", str(ROOT), root],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("M3 policy requires the exact lexical repository tuple", result.stderr)

    def test_copied_tuple_cannot_select_generic_or_predecessor_policy(self):
        names = ("root.esk", "bridge.c", "private_renames.txt", "public_exports.txt")
        exact = [str(ROOT / f"native/m3_package_{name}") for name in names]
        includes = [str(ROOT / name) for name in ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib", "src")]
        with tempfile.TemporaryDirectory() as directory:
            copied = []
            for index, source in enumerate(exact):
                path = Path(directory) / f"copy-{index}"
                path.write_text(Path(source).read_text() + "\n")
                copied.append(str(path))
            for inputs, roots in ((copied, includes),
                                  ([exact[0], str(ROOT / "native/m3t_package_bridge.c"), *exact[2:]], includes),
                                  (exact, list(reversed(includes)))):
                result = subprocess.run(["/usr/bin/bash", str(ROOT / "scripts/build-e1b-consumer.sh"),
                    *inputs, str(Path(directory) / "out.o"), *roots], cwd=ROOT, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("M3 policy", result.stderr)
                self.assertFalse((Path(directory) / "out.o").exists())


if __name__ == "__main__":
    unittest.main()
