"""Source shape for installed G2 caller and conditional F-copy attestation."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/g3c4_public_g2_result_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_public_g2_f_artifact_source_closure.txt"
ADDED = (
    "tests/g3g_manual/public_g2_runtime.esk",
    "tests/g3c4/f_artifact_attest.py",
    "tests/g3c4/check_public_g2_f_artifact.py",
    "tests/g3c4/test_f_artifact_attest.py",
    "docs/g3/G3_C4_PUBLIC_G2_F_ARTIFACT_SOURCE_LEAF.md",
    "native/g3c4_public_g2_f_artifact_source_closure.txt",
)
UNCHANGED = {
    "native/g3c4_public_g2_result_extension.esk":
        "8a3cda9fbe067ff50785977d26a647a7e6e5b2163a508560f59d33c2555b5458",
    "native/g3g_manual_public_extension.esk":
        "f5f0cdd162bcecb7b62c57fc4ce0dda593de3926e32061e8c8d2c9e715e7c891",
    "tests/g3c4/p2_g2_public_result_test.esk":
        "10c58fd0885f0098dc57990fea596c664915a522ee919002edda4b34612ff0be",
    "tests/g3c4/p2_g1_public_generation_test.esk":
        "ddabecc6fa9fa7fc094f76f3bf6c0a03d84dee1c785fc0d14c8519e698381d4e",
    "scripts/common.sh":
        "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83",
    "toolchain/eshkol.lock":
        "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe",
}


def validate(fixture: str, attester: str, base: list[str],
             closure: list[str], source: dict[str, bytes]) -> None:
    assert len(base) == 216 and closure == base + list(ADDED)
    assert len(closure) == len(set(closure))
    assert all((ROOT / member).is_file() for member in closure)
    for path, expected in UNCHANGED.items():
        assert hashlib.sha256(source[path]).hexdigest() == expected, path
    ast.parse(attester)
    assert fixture.count("(require transformer.generation)") == 1
    assert fixture.count("(require transformer.diagnostic_transport)") == 1
    assert "(load " not in fixture and "g3c4-registry" not in fixture
    assert "g3c4-p2g2-terminal" not in fixture
    assert "et_e1b_" not in fixture and "(extern " not in fixture
    for text in ("diagnostic-c4-model-create-seeded", "generation-c4-input-create",
                 "generator-create", "generator-generate!", "generation-output-ids",
                 "generation-output-lengths", "generation-output-cache-lengths",
                 "generation-output-text", "generation-output-rng",
                 "generation-output-release!", "generation-tensor-release!",
                 "generation-rng-release!", "generator-close!",
                 "':profile 'diagnostic-c4-p2g2", "':max-new-tokens 2",
                 "first EOS has one raw byte", "non-EOS has two raw bytes",
                 "G3G-MANUAL-PUBLIC-G2-PASS"):
        assert text in fixture, text
    assert fixture.count("(generation-tensor-release! prompt)") == 1
    assert fixture.count("(generation-rng-release! eos-rng)") == 1
    assert fixture.count("(generation-rng-release! two-rng)") == 1
    assert fixture.index("(generator-close! eos-generator)") < fixture.index(
        "first EOS text survives source close")
    assert fixture.index("(generator-close! two-generator)") < fixture.index(
        "non-EOS detached owners release")
    assert "historical-F-remains-failed" in attester
    for text in ("COMMON_TRANSFORMS", "original_output(", "project_copy(",
                 "copied_artifacts(", "final_copied_artifacts(",
                 "resolved_image_libs(", "verify_prepared(", "source_admission(",
                 "candidate_lock(", "exclusive_bytes(",
                 "new-test-only-F-copy;not-bootstrap"):
        assert text in attester, text
    assert "--project-head" not in attester and "--project-tree" not in attester
    assert "unclassified filesystem node" in attester
    assert attester.count('"/bundle/public-g2-6803.lock"') >= 2
    assert "ESHKOL_ALLOW_UNSUPPORTED_HOST" in attester


def main() -> None:
    source = {path: (ROOT / path).read_bytes() for path in UNCHANGED}
    validate((ROOT / ADDED[0]).read_text(), (ROOT / ADDED[1]).read_text(),
             BASE.read_text().splitlines(), CLOSURE.read_text().splitlines(),
             source)
    print("public G2 F-artifact source contract: PASS")


if __name__ == "__main__":
    main()
