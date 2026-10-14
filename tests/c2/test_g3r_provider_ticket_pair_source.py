"""Cross-language contract for the private provider-ticket pair migration."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / "native/g3r_same_owner_hold_bridge.c"
PAIR_HEADER = ROOT / "native/g3r_same_owner_hold_bridge.h"
PAIR_ESK = ROOT / "native/g3r_same_owner_hold_bridge_extension.esk"
ADOPTER = ROOT / "native/o2_wave2_package_bridge.c"
ADOPTER_ESK = ROOT / "native/g3r_held_o2_adopter_extension.esk"


PAIR_EDGES = (
    "pair->g3_ticket = et_g3r_private_g3t_save_begin_v2(",
    "&pair->o2_ticket, &error);",
    "et_g3r_private_g3t_save_check_v2(pair->g3_ticket)",
    "et_g3r_private_o2_save_owner_match_v2(",
    "et_g3r_private_o2_save_check_v2(pair->o2_ticket, &error)",
    "et_g3r_private_g3t_save_copy_v2(",
    "et_g3r_private_o2_save_end_v2(",
    "et_g3r_private_g3t_save_end_v2(pair->g3_ticket)",
    "if (pair->o2_ticket != 0) abort();",
    "*slot = pair->next;",
    "free(pair);",
)
FFI_EDGES = (
    ":real et_g3r_private_same_owner_holds_begin_v2",
    ":real et_g3r_private_same_owner_holds_check_v2",
    "(extern i64 g3r-holds-native-g3 i64",
    "(extern i64 g3r-holds-native-o2 i64",
    ":real et_g3r_private_same_owner_holds_copy_g3_v2",
    ":real et_g3r_private_same_owner_holds_end_v2",
    ":real et_g3r_private_o2_adopter_owner_match_v2",
    "(extern ptr g3r-held-o2-issue i64",
    "(extern i64 g3r-held-o2-issued-match i64 ptr ptr ptr",
)


def validate(pair, header, pair_esk, adopter, adopter_esk):
    for edge in PAIR_EDGES:
        if edge not in pair:
            raise ValueError(f"pair ticket edge missing: {edge}")
    if "int64_t et_g3r_private_same_owner_holds_g3_v2" not in header or \
            "int64_t et_g3r_private_same_owner_holds_o2_v2" not in header or \
            "void *et_g3r_private_same_owner_holds_g3_v1" not in header:
        raise ValueError("pair ticket and preserved pointer ABIs differ")
    for edge in FFI_EDGES[:6]:
        if edge not in pair_esk:
            raise ValueError(f"pair Eshkol FFI mismatch: {edge}")
    for edge in FFI_EDGES[6:]:
        if edge not in adopter_esk:
            raise ValueError(f"adopter Eshkol FFI mismatch: {edge}")
    adopter_v2 = adopter.split(
        "int64_t et_g3r_private_o2_adopter_owner_match_v2(", 1)[1]
    adopter_v2 = adopter_v2.split(
        "/* An indeterminate native lifecycle", 1)[0]
    if "issued == NULL || issued != candidate" not in adopter_v2 or \
            "et_g3r_private_o2_save_issued_match_v2(" not in adopter_v2:
        raise ValueError("exact issued-state pointer check removed")


class ProviderTicketPairSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = (PAIR.read_text(), PAIR_HEADER.read_text(),
                    PAIR_ESK.read_text(), ADOPTER.read_text(),
                    ADOPTER_ESK.read_text())

    def test_contract(self):
        validate(*self.args)

    def test_mutations(self):
        for index, edges in ((0, PAIR_EDGES), (2, FFI_EDGES[:6]),
                             (4, FFI_EDGES[6:])):
            for edge in edges:
                with self.subTest(edge=edge), self.assertRaises(ValueError):
                    args = list(self.args)
                    args[index] = args[index].replace(edge, "/* removed */")
                    validate(*args)
        with self.assertRaises(ValueError):
            args = list(self.args)
            args[3] = args[3].replace(
                "issued == NULL || issued != candidate",
                "issued == NULL || issued == candidate")
            validate(*args)


if __name__ == "__main__":
    unittest.main()
