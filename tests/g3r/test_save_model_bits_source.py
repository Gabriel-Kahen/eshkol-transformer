"""Source contract for the test-only live-model bit images around the cut."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.test_same_owner_save_caller_source import definition


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tests/g3r/save_post_adoption_cut_runtime.esk"
FIXTURE_ROOT = ROOT / "tests/g3r/save_post_adoption_cut_root.esk"
BRIDGE = ROOT / "tests/g3r/save_model_bits_bridge.c"
F32_HEADER = ROOT / "native/f32_parameter_internal.h"
F32_SOURCE = ROOT / "native/f32_tensor.c"


def validate(runtime: str, root: str, bridge: str) -> None:
    forms = parse(runtime)
    values = str(definition(runtime, ["capture-value-images"]))
    gradients = str(definition(runtime, ["capture-gradient-images"]))
    compare = str(definition(runtime, ["same-images?", "before", "after"]))
    tie = str(definition(runtime, ["original-tie?"]))
    for fragment in ("['=', 'index', '14']", "['vector-ref', 'model-handles', 'index']",
                     "i2-checkpoint-encode", "parameter-handle-tensor-internal",
                     "parameter-handle-shape"):
        assert fragment in values, fragment
    for fragment in ("['=', 'index', '14']", "gradient-byte-lengths",
                     "g3r-save-cut-gradient-copy", "model-native", "index",
                     "['vector-set!', 'images', 'index', 'image']"):
        assert fragment in gradients, fragment
    assert "d2-core-bytes=?" in compare and compare.count("'14'") >= 3
    assert "'#(64 64 64 64 128 128 16 16 16 16 4096 16 16 32)" in runtime
    for fragment in ("parameter-tree-tie-groups", "parameter-tree-handle",
                     "token_embedding", "head", "['vector-ref', 'model-handles', '10']"):
        assert fragment in tie, fragment
    assert tie.count("eq?") >= 2

    def index_of(text: str) -> int:
        position = next((i for i, form in enumerate(forms)
                         if text in str(form)), -1)
        assert position >= 0, text
        return position

    assert (index_of("original tied alias before cut") <
            index_of("value-images-before") <
            index_of("gradient-images-before") <
            index_of("typed post-adoption failure") <
            index_of("fourteen original value bit images unchanged") <
            index_of("fourteen original gradient bit images unchanged") <
            index_of("original tied alias after cut"))
    assert "same-images? value-images-before (capture-value-images)" in runtime
    assert "same-images? gradient-images-before (capture-gradient-images)" in runtime
    assert "(original-tie?)" in runtime
    assert "g3r-save-cut-seed-gradients" not in root + runtime
    assert ":real et_g3r_test_save_gradient_copy_v1" in root
    for fragment in ("defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE)",
                     "defined(ET_F32_TENSOR_TESTING)",
                     "defined(ET_G3R_SAVE_BIT_WITNESS_TESTING)",
                     "words[14]", "16, 16, 16, 16, 32, 32, 4, 4, 4, 4, 1024, 4, 4, 8",
                     "et_m3t_private_owner_save_roster_v1",
                     "et_f32_parameter_gradient_metadata_v1",
                     "metadata.state != ET_F32_GRADIENT_ABSENT",
                     "metadata.contribution_count != 0",
                     "metadata.normalization_weight_bits != 0",
                     "et_f32_parameter_test_gradient_tensor_v1",
                     "et_f32_tensor_copy_bits_to_v1",
                     "destination[i] != UINT32_C(0)",
                     "index < 0 || index >= 14", "encoded != byte_count"):
        assert fragment in bridge, fragment
    assert "m3_last_owner" not in bridge
    for fragment in ("et_f32_parameter_test_set_gradient_bits_v1",
                     "et_f32_parameter_test_set_metadata_v1",
                     "et_f32_parameter_gradient_snapshot_v1"):
        assert fragment not in bridge, fragment
    header, source = F32_HEADER.read_text(), F32_SOURCE.read_text()
    for native in (header, source):
        assert "defined(ET_G3R_SAVE_BIT_WITNESS_TESTING)" in native
        assert "et_f32_parameter_test_gradient_tensor_v1" in native


class SaveModelBitsSource(unittest.TestCase):
    def test_exact_owner_witness_contract(self):
        validate(RUNTIME.read_text(), FIXTURE_ROOT.read_text(), BRIDGE.read_text())

    def test_missing_or_misdirected_witness_rejected(self):
        runtime, root, bridge = (RUNTIME.read_text(), FIXTURE_ROOT.read_text(),
                                 BRIDGE.read_text())
        mutations = (
            (runtime.replace("(= index 14) images", "(= index 13) images", 1), root, bridge),
            (runtime.replace("model-native index image bytes", "optimizer index image bytes", 1), root, bridge),
            (runtime.replace("4096 16 16 32", "4092 16 16 32", 1), root, bridge),
            (runtime.replace('(check "fourteen original gradient bit images unchanged"',
                             '(check "gradient unchecked"', 1), root, bridge),
            (runtime.replace('(check "original tied alias after cut"',
                             '(check "tie unchecked"', 1), root, bridge),
            (runtime, root.replace("et_g3r_test_save_gradient_copy_v1", "wrong_symbol", 1), bridge),
            (runtime, root, bridge.replace("et_m3t_private_owner_save_roster_v1", "untrusted_roster", 1)),
            (runtime, root, bridge.replace("et_f32_tensor_copy_bits_to_v1", "approximate_copy", 1)),
            (runtime, root, bridge.replace("index < 0 || index >= 14", "index < 0 || index >= 15", 1)),
            (runtime, root, bridge.replace("metadata.state != ET_F32_GRADIENT_ABSENT",
                                           "metadata.state != ET_F32_GRADIENT_PRESENT", 1)),
            (runtime, root, bridge.replace("metadata.contribution_count != 0",
                                           "metadata.contribution_count != 1", 1)),
            (runtime, root, bridge.replace("metadata.normalization_weight_bits != 0",
                                           "metadata.normalization_weight_bits != UINT32_C(0x3f800000)", 1)),
            (runtime, root, bridge.replace("destination[i] != UINT32_C(0)",
                                           "destination[i] != UINT32_C(0x80000000)", 1)),
            (runtime, root, bridge.replace("return gradient_copy(parameters[index]",
                                           "et_f32_parameter_test_set_metadata_v1(parameters[index], 1, 1, 1);\n  return gradient_copy(parameters[index]", 1)),
        )
        for changed in mutations:
            with self.subTest(changed=str(changed)[:50]), self.assertRaises(AssertionError):
                validate(*changed)


if __name__ == "__main__":
    unittest.main()
