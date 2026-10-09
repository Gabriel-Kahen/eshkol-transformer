"""Owned source controls for the staged public C4 P2/G2 result adapter."""

from __future__ import annotations

import hashlib
from importlib.machinery import SourceFileLoader
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
parse = SourceFileLoader(
    "g2_result_source_shape",
    str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py"),
).load_module()
BASE = ROOT / "native/g3c4_public_g2_ticket_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_public_g2_result_source_closure.txt"
ADDITIONS = [
    "scripts/build-e1b-consumer.sh",
    "native/g3c4_public_g2_result_extension.esk",
    "native/g3g_manual_public_extension.esk",
    "tests/g3c4/p2_g2_public_result_test.esk",
    "tests/g3c4/test_public_g2_result_source.py",
    "docs/g3/G3_C4_PUBLIC_G2_RESULT_SOURCE_LEAF.md",
    "native/g3c4_public_g2_result_source_closure.txt",
]
ADAPTER_SHA256 = "3d55f8737b6fcfc280a9c4094362d69eaa1d005002cf1b604f2464d6f40d2139"
DISPATCH_SHA256 = "1a3918c988c8ffd0c236ba8899e22643a02a9498cb45874b0abf998d6cd03895"
P2G1_FIXTURE_SHA256 = "ddabecc6fa9fa7fc094f76f3bf6c0a03d84dee1c785fc0d14c8519e698381d4e"
FLAGS = {
    "ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE",
    "ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE",
    "ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE",
    "ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE",
    "ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE",
    "ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE",
    "ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE",
}


def definition(source: str, name: str) -> list[object]:
    matches = [form for form in parse.forms(source)
               if isinstance(form, list) and form and form[0] == "define"
               and isinstance(form[1], list) and form[1][0] == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def validate(dispatch: bytes, result: str, manual: str, package_root: str,
             package_sources: list[str], package_checker: str,
             ticket_checker: str, build: str, fixture: str,
             adapter: bytes, p2g1_fixture: bytes,
             closure: list[str], base: list[str],
             header: str, owner: str) -> None:
    assert closure == base + ADDITIONS
    assert len(closure) == len(set(closure))
    assert all((ROOT / member).is_file() for member in closure)
    assert hashlib.sha256(adapter).hexdigest() == ADAPTER_SHA256
    assert hashlib.sha256(dispatch).hexdigest() == DISPATCH_SHA256
    assert hashlib.sha256(p2g1_fixture).hexdigest() == P2G1_FIXTURE_SHA256
    assert package_sources[-3:] == [
        "native/g3c4_public_generation_dispatch_extension.esk",
        "native/g3c4_public_g2_result_extension.esk",
        "native/g3g_manual_public_extension.esk",
    ]
    assert len(package_sources) == 57
    assert "len(sources) == 57" in package_checker
    assert "sources[-18:]" in package_checker
    assert '"g3c4_public_g2_result_extension"' in package_checker
    assert "len(package_sources) == 57" in ticket_checker
    order = [f'(load "{Path(member).name}")' for member in package_sources[-3:]]
    assert all(package_root.count(item) == 1 for item in order)
    assert [package_root.index(item) for item in order] == sorted(
        package_root.index(item) for item in order)

    output_live = definition(result, "g3c4-public-g2-output-live")
    assert len(parse.calls(output_live, "g3c4-p2g2-terminal-origin-live")) == 1
    assert len(parse.calls(output_live, "g3c4-public-g2-profile-original")) == 1
    assert len(parse.calls(output_live, "g3c4-public-g2-output-unique")) == 4
    assert "((vector-ref binding 1) 'verify row anchor entry" in result
    output_release = definition(result, "g3c4-public-g2-output-release!")
    assert len(parse.calls(output_release, "g3c4-p2g2-terminal-release-private!")) == 1
    assert result.index("(g3c4-p2g2-terminal-release-private! shell)",
                        result.index("(define (g3c4-public-g2-output-release!")) < result.index(
        "(vector-set! row 8 'dead)",
        result.index("(define (g3c4-public-g2-output-release!"))

    source_snapshot = definition(result, "g3c4-public-g2-source-snapshot")
    assert len(parse.calls(source_snapshot, "g3c4-public-g2-output-live")) == 1
    assert len(parse.calls(source_snapshot,
                           "g3c4-p2g2-terminal-snapshot-private")) == 1
    resnapshot = definition(result, "g3c4-public-g2-resnapshot")
    assert len(parse.calls(resnapshot, "g3c4-p2g2-native-snapshot")) == 1
    assert len(parse.calls(resnapshot, "g3c4-public-g2-snapshot-equal?")) == 1
    clone = definition(result, "g3c4-public-g2-detached-ids")
    for call in ("g3c4-p2g2-terminal-ids-clone-private",
                 "g3c4-p2g2-terminal-ids-clone-copy-private",
                 "g3c4-p2g2-terminal-ids-clone-release-private!"):
        assert parse.calls(clone, call)
    publish = definition(result, "g3c4-public-g2-result-publish")
    assert len(parse.calls(publish, "g3c4-native-terminal-result-create")) == 1
    assert len(parse.calls(publish, "g3c4-public-g2-resnapshot")) == 1
    assert len(parse.calls(publish, "g3c4-native-result-rng-create")) == 1
    assert "(eq? (vector-ref candidate-entry 10) ledger)" in result
    assert parse.calls(publish, "g3c4-native-result-tensor-release")
    assert parse.calls(publish, "g3c4-native-rng-release")
    assert "(vector-set! entry 2 'live)" in result
    result_release = definition(result, "g3c4-public-g2-result-release!")
    assert len(parse.calls(result_release, "g3c4-public-g2-result-live")) == 1
    assert parse.calls(result_release, "g3c4-native-result-tensor-release")
    assert parse.calls(result_release, "g3c4-native-rng-release")
    release_text = result[result.index("(define (g3c4-public-g2-result-release!"):
                          result.index("(define (g3c4-public-g2-source-snapshot")]
    assert "(eq? (vector-ref entry 3) #f)" in release_text
    assert "(eq? (vector-ref entry 10) #f)" in release_text
    output_release_text = result[
        result.index("(define (g3c4-public-g2-output-release!"):
        result.index(";;; Detached G2 results")]
    assert "g3c4-public-g2-tickets 5 shell operation" in output_release_text
    output_access = definition(result, "g3c4-public-g2-output-access")
    assert len(parse.calls(output_access, "g3c4-p2g2-terminal-live")) == 1
    assert "(vector-ref terminal 7)" in result
    assert "(vector-ref terminal 8)" in result
    assert "(vector-ref source 7)" not in str(output_access)

    for public in ("ids", "lengths", "text", "rng", "cache-lengths"):
        source = definition(manual, f"g3g-public-generation-output-{public}")
        assert len(parse.calls(source, "g3c4-public-g2-output-access")) == 1
        assert ["eq?", ["vector-ref", "entry", "1"],
                "'terminal-output"] in parse.calls(source, "eq?")
        assert len(parse.calls(source,
                               f"g3c4-public-generation-output-{public}")) == 1
        assert f"g3t-generation-output-{public}" in str(source)
    for public, private in (
        ("g3g-public-generation-output-release", "g3c4-public-g2-output-release!"),
        ("g3g-public-generation-tensor-release", "g3c4-public-g2-result-release!"),
        ("g3g-public-generation-rng-release", "g3c4-public-g2-result-release!"),
    ):
        assert len(parse.calls(definition(manual, public), private)) == 1

    owner_start = build.index('elif [[ "${package_policy}" == g3g-manual-public-aggregate &&\n'
                              '            "${package_native_source}" == '
                              '"${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" ]]')
    owner_end = build.index("\n    fi\n", owner_start)
    owner_flags = re.findall(r"-D(ET_[A-Z0-9_]+)", build[owner_start:owner_end])
    assert len(owner_flags) == len(set(owner_flags))
    assert set(owner_flags) & FLAGS == FLAGS
    a2_start = build.index('    elif [[ "${g3g_package_policy}" == 1 &&\n'
                           '            "${package_native_source}" == '
                           '"${PROJECT_ROOT}/native/a2_kv_cache.c" ]]')
    a2_end = build.index("\n    elif", a2_start)
    a2 = build[a2_start:a2_end]
    assert a2.count("-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE") == 1
    assert 'if [[ "${package_policy}" == g3g-manual-public-aggregate ]]' in a2
    assert set(re.findall(r"-D(ET_G3C4_P2_G2_[A-Z0-9_]+)", build)) == FLAGS - {
        "ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE"}
    for flag in FLAGS - {"ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE"}:
        assert build.count("-D" + flag) == 1
    assert build.count("-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE") == 2
    assert build.count('"${package_source_flags[@]}"') == 1
    assert ('"${e1b_cc}" "${package_native_cflags[@]}" '
            '"${package_source_flags[@]}"') in build
    assert "#ifdef ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE" in header
    assert "et_g3c4_private_terminal_result_tensor_create_v1(" in header
    assert "#ifdef ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE" in owner
    assert "void *et_g3c4_private_terminal_result_tensor_create_v1(\n" in owner
    assert "int64_t kind, int64_t count, int64_t value0, int64_t value1" in owner

    for marker in (
        'out-of-band private origin does not gain public access',
        'missing public origin cannot fall back to C2',
        'missing private origin cannot publish',
        'paired public creator substitution cannot forge authority',
        'reused pending ticket cannot publish',
        'changed first-EOS terminal text blocks public accessor',
        'changed non-EOS terminal text blocks public accessor',
        'genuine first EOS stops at one ID',
        'genuine non-EOS takes two IDs',
        'rejected release retains native output',
        'dead output with residual ticket root rejects',
        'dead I1 result with reused native pointer rejects',
        'dead I1 result with reused ledger rejects',
        'dead I1 result with changed owner value rejects',
        'missing RNG owner binding blocks native retirement',
        'same-native replacement RNG ledger rejects',
        'detached IDs release after source close',
        'detached RNG release after source close',
    ):
        assert marker in fixture


class PublicG2ResultSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        read = lambda path: (ROOT / path).read_text()
        cls.inputs = dict(
            dispatch=(ROOT / "native/g3c4_public_generation_dispatch_extension.esk").read_bytes(),
            result=read("native/g3c4_public_g2_result_extension.esk"),
            manual=read("native/g3g_manual_public_extension.esk"),
            package_root=read("native/g3g_manual_package_root.esk"),
            package_sources=read("native/g3g_manual_package_source_closure.txt").splitlines(),
            package_checker=read("scripts/check-g3g-manual-package.py"),
            ticket_checker=read("scripts/check-g3c4-public-g2-ticket.py"),
            build=read("scripts/build-e1b-consumer.sh"),
            fixture=read("tests/g3c4/p2_g2_public_result_test.esk"),
            adapter=(ROOT / "native/g3c4_public_output_adapter_extension.esk").read_bytes(),
            p2g1_fixture=(ROOT / "tests/g3c4/p2_g1_public_generation_test.esk").read_bytes(),
            closure=CLOSURE.read_text().splitlines(),
            base=BASE.read_text().splitlines(),
            header=read("src/eshkol_transformer/g3c4_context_internal.h"),
            owner=read("src/eshkol_transformer/g3c4_model_owner.c"),
        )

    def reject(self, **changes: object) -> None:
        candidate = dict(self.inputs)
        candidate.update(changes)
        with self.assertRaises(AssertionError):
            validate(**candidate)

    def test_source(self) -> None:
        validate(**self.inputs)

    def test_missing_public_and_private_origin(self) -> None:
        start = self.inputs["result"].index(
            "(define (g3c4-public-g2-output-live")
        for name in ("g3c4-p2g2-terminal-origin-live",
                     "g3c4-public-g2-profile-original"):
            self.reject(result=self.inputs["result"][:start] +
                        self.inputs["result"][start:].replace(
                            f"({name}", "(missing-auth", 1))

    def test_no_snapshot_or_clone_shortcut(self) -> None:
        for name in ("g3c4-p2g2-terminal-snapshot-private",
                     "g3c4-p2g2-terminal-ids-clone-copy-private",
                     "g3c4-p2g2-native-snapshot"):
            self.reject(result=self.inputs["result"].replace(
                f"({name}", "(missing-copy", 1))
        self.reject(result=self.inputs["result"].replace(
            "(vector-ref terminal 7)", "(vector-ref source 7)", 1))

    def test_missing_owner_and_cleanup(self) -> None:
        for name in ("g3c4-native-terminal-result-create",
                     "g3c4-native-result-tensor-release"):
            self.reject(result=self.inputs["result"].replace(
                f"({name}", "(missing-owner", 1))
        start = self.inputs["result"].index(
            "(define (g3c4-public-g2-result-release!")
        self.reject(result=self.inputs["result"][:start] +
                    self.inputs["result"][start:].replace(
                        "(eq? (vector-ref entry 3) #f)", "#t", 1))
        self.reject(result=self.inputs["result"].replace(
            "(eq? (vector-ref candidate-entry 10) ledger)", "#t", 1))

    def test_no_public_fallback_or_missing_fixture(self) -> None:
        start = self.inputs["manual"].index(
            "(define (g3g-public-generation-output-ids")
        self.reject(manual=self.inputs["manual"][:start] +
                    self.inputs["manual"][start:].replace(
                        "(eq? (vector-ref entry 1) 'terminal-output)",
                        "#f", 1))
        self.reject(fixture=self.inputs["fixture"].replace(
            "out-of-band private origin does not gain public access",
            "missing private denial", 1))

    def test_wrong_tuple_and_partial_flags(self) -> None:
        self.reject(build=self.inputs["build"].replace(
            "-DET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE", "", 1))
        self.reject(build=self.inputs["build"].replace(
            'if [[ "${package_policy}" == g3g-manual-public-aggregate ]]; then\n'
            '        package_source_flags+=(-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)',
            'if [[ "${package_policy}" == g3g-g0-public-aggregate ]]; then\n'
            '        package_source_flags+=(-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)', 1))
        self.reject(build=self.inputs["build"].replace(
            "-DET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE",
            "-DET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE "
            "-DET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE", 1))
        self.reject(build=self.inputs["build"].replace(
            "-DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE", "", 1))
        self.reject(build=self.inputs["build"].replace(
            "-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE", "", 1))
        self.reject(build=self.inputs["build"].replace(
            "-DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE",
            "-DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE "
            "-DET_G3C4_P2_G2_EXTRA_PRIVATE", 1))

    def test_wrong_closure_or_old_adapter_drift(self) -> None:
        self.reject(closure=self.inputs["closure"][:-1])
        self.reject(adapter=self.inputs["adapter"] + b"\n;; drift\n")
        self.reject(dispatch=self.inputs["dispatch"] + b"\n;; drift\n")
        self.reject(p2g1_fixture=self.inputs["p2g1_fixture"] + b"\n;; drift\n")
        self.reject(package_sources=self.inputs["package_sources"][:-1])
        self.reject(package_checker=self.inputs["package_checker"].replace(
            "len(sources) == 57", "len(sources) == 56", 1))
        self.reject(ticket_checker=self.inputs["ticket_checker"].replace(
            "len(package_sources) == 57", "len(package_sources) == 56", 1))
        self.reject(package_root=self.inputs["package_root"].replace(
            '(load "g3c4_public_g2_result_extension.esk")\n', "", 1))


if __name__ == "__main__":
    unittest.main()
