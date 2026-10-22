/* Exercise the exact bridge used by the compiled Eshkol diagnostic. */
#include "private_terminal_owners_native_bridge.c"
extern int64_t et_m3_test_i64_counts_v1(int64_t field);

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  const int64_t prompt[2] = {7, 11};
  for (int first_eos = 0; first_eos < 2; ++first_eos) {
    et_g3c4_context_internal *context;
    et_g3c4_input_internal *input;
    et_g3c4_output_internal *output;
    terminal_raw first, second;
    terminal_ready(owner, prompt, first_eos, &context, &input, &output,
                   &first, &second);
    OK(et_g3c4_private_p2g2_terminal_commit_v1(
        context, output, &first, 9, first_eos ? NULL : &second,
        first_eos ? 0 : 9));
    void *clone = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
    CHECK(clone != NULL);
    et_i64_tensor *original_ids = output->ids;
    CHECK(et_g3c4_private_probe_output_live_v1(NULL) == -1);
    CHECK(et_g3c4_private_probe_output_live_v1((void *)(uintptr_t)1) == -1);
    CHECK(et_g3c4_private_probe_output_live_v1(input) == -1);
    CHECK(et_g3c4_private_probe_output_live_v1(clone) == -1);
    uint64_t magic = output->transport.magic;
    output->transport.magic = 0;
    CHECK(et_g3c4_private_probe_output_live_v1(output) == -1);
    output->transport.magic = magic;
    int64_t borrows_before = et_m3_test_i64_counts_v1(1);
    OK(et_g3c4_private_probe_output_hold_v1(output));
    CHECK(held_output_ids != NULL);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before + 1);
    CHECK(et_g3c4_private_output_release_v1(output) != 0);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before + 1);
    CHECK(output->ids == original_ids);
    OK(et_g3c4_private_probe_output_live_v1(output));
    OK(et_g3c4_private_probe_output_end_v1());
    CHECK(held_output_ids == NULL);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before);
    OK(et_g3c4_private_output_release_v1(output));
    CHECK(et_g3c4_private_probe_output_live_v1(output) == 1);
    CHECK(output->ids == NULL);

    /* Attribute E's clone lease/release boundary independently of Eshkol. */
    et_g3c4_terminal_ids_clone_internal *registered_clone =
        et_g3c4_terminal_ids_admit(clone, 0);
    CHECK(registered_clone != NULL);
    original_ids = registered_clone->ids;
    OK(et_g3c4_private_probe_clone_hold_v1(clone));
    CHECK(held_clone_ids != NULL);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before + 1);
    CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone) != 0);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before + 1);
    CHECK(registered_clone->ids == original_ids);
    OK(et_g3c4_private_probe_clone_live_v1(clone));
    OK(et_g3c4_private_probe_clone_end_v1());
    CHECK(held_clone_ids == NULL);
    CHECK(et_m3_test_i64_counts_v1(1) == borrows_before);
    OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone));
    CHECK(et_g3c4_private_probe_clone_live_v1(clone) == 1);
    CHECK(registered_clone->ids == NULL);
    OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone));
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  printf("G3-C4 held terminal/clone probe PASS: checks=%zu\n", checks);
  return 0;
}
