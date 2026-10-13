"""Source pins for the proposed current-history held SAVE contract.

These tests check current owners and causal drift only. They do not compile or
exercise a held owner, which has not been implemented.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = (ROOT / "src/eshkol_transformer/g3t_transport.h").read_text()
NATIVE = (ROOT / "src/eshkol_transformer/g3t_transport.c").read_text()
SNAPSHOT = (ROOT / "native/g3r_committed_save_snapshot_extension.esk").read_text()
C2 = (ROOT / "native/c2_training_state_extension.esk").read_text()


def current_history_contract(header: str, native: str) -> bool:
    structure = header.split("typedef struct et_g3r_committed_history_snapshot_v1", 1)[-1].split("}", 1)[0]
    capture = native.split("int64_t et_g3t_private_committed_history_snapshot_v1(", 1)[-1].split("#ifdef ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE", 1)[0]
    required = (
        "int64_t policy[6], rng[4], ids[2];",
        "int64_t length, prompt_length, generated_count, manual_count;",
        "int64_t generated_eos_terminal;",
    )
    copies = (
        "memcpy(snapshot.policy, c->policy, sizeof(snapshot.policy));",
        "memcpy(snapshot.rng, c->rng, sizeof(snapshot.rng));",
        "memcpy(snapshot.ids, c->history_ids, sizeof(snapshot.ids));",
        "snapshot.length = c->history_length;",
        "snapshot.prompt_length = c->history_prompt_length;",
        "snapshot.generated_count = c->history_generated_count;",
        "snapshot.manual_count = c->history_manual_count;",
        "snapshot.generated_eos_terminal = c->history_generated_eos_terminal;",
    )
    return (
        "ET_G3R_COMMITTED_HISTORY_PRIVATE" in header
        and "ET_G3R_G3T_SAVE_LEDGER_PRIVATE" not in header
        and all(item in structure for item in required)
        and all(item in capture for item in copies)
        and "c->history_pending_g1 ||" in capture
        and "!g3t_history_valid(c)" in capture
        and "sizeof(et_g3r_committed_history_snapshot_v1) == 136" in native
    )


class CurrentHistoryHeldContractSource(unittest.TestCase):
    def test_current_owner_layout_and_pending_g1_denial(self):
        self.assertTrue(current_history_contract(HEADER, NATIVE))

    def test_pending_g1_removal_is_detected(self):
        self.assertFalse(current_history_contract(
            HEADER, NATIVE.replace("c->history_pending_g1 ||", "")))

    def test_history_and_terminal_substitution_is_detected(self):
        self.assertFalse(current_history_contract(
            HEADER, NATIVE.replace("snapshot.manual_count = c->history_manual_count;",
                                   "snapshot.manual_count = 0;", 1)))
        self.assertFalse(current_history_contract(
            HEADER, NATIVE.replace("snapshot.generated_eos_terminal = c->history_generated_eos_terminal;",
                                   "snapshot.generated_eos_terminal = 0;", 1)))

    def test_old_eighteen_word_layout_is_not_accepted(self):
        self.assertFalse(current_history_contract(
            HEADER.replace("int64_t policy[6], rng[4], ids[2];", "int64_t word[18];", 1), NATIVE))

    def test_copyout_is_transient_and_tokenizer_is_separate(self):
        body = NATIVE.split("int64_t et_g3t_private_committed_save_copy_v1(", 1)[1]
        self.assertIn("et_g3t_model_pins_end_internal(&pins);", body)
        self.assertIn("et_a2_kv_cache_read_borrow_end_v1(&borrow, &error)", body)
        self.assertIn("(define (g3r-save-tokenizer? tokenizer)", SNAPSHOT)
        self.assertIn("(eq? (vector-ref core 1) 'raw)", SNAPSHOT)
        self.assertIn("(= (t1-tokenizer-core-vocab-size core) 256)", SNAPSHOT)
        self.assertIn("(vector (vector-ref state 1) (vector-ref state 2)", C2)
        self.assertIn("(vector-ref state 3) (vector-ref state 4)", C2)


if __name__ == "__main__":
    unittest.main()
