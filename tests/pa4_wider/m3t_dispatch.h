/* Included only by the native M3T diagnostic build. */
#include "eshkol_transformer/pa4_wider_attention_abi.h"

static int select_pa4_wider_reference;
static unsigned attention_resolutions;

const et_kernel_provider_v1 *__real_et_a2_kernel_provider_v1(void);
const et_kernel_provider_v1 *__wrap_et_a2_kernel_provider_v1(void) {
  ++attention_resolutions;
  return select_pa4_wider_reference ? et_pa4_wider_ref_provider_v1()
                                   : __real_et_a2_kernel_provider_v1();
}

static void check_attention_provider(void) {
  const uint64_t shape[] = {1, 2, 2, 2, 2, 2};
  const char *operations[] = {"causal-attention.forward", "causal-attention.backward"};
  const char *implementation = select_pa4_wider_reference
      ? "pa4-wider.cpu-f32.reference" : "a2.cpu-f32.serial";
  CHECK(attention_resolutions == 1);
  for (size_t i = 0; i < 2; ++i) {
    et_kernel_request_v1 request = {
        sizeof(request), operations[i], "f32", "cpu", 6, shape, 1, {0}};
    const et_kernel_capability_v1 *capability = NULL;
    et_kernel_error error;
    OK(et_kernel_runtime_capability_require(
        runtimes[2], "kernel.causal-attention", &request, &capability, &error));
    CHECK(capability && !strcmp(capability->implementation, implementation));
  }
  fprintf(stderr, "M3T selected %s for forward and backward\n", implementation);
}

static void reject_invalid_attention_mask(workspace *w, int reverse_phase,
                                           frame_copy *before) {
  et_f32_test_live_counts_v1 live = live_counts();
  CHECK(w->keep[0] == 1);
  w->keep[0] = 2;
  capture(w, before);
  CHECK(et_m3t_private_attention_v1(w, reverse_phase) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error_domain == 1 && error_category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
        error_code == ET_KERNEL_CODE_INVALID_BUFFER);
  CHECK(w->keep[0] == 2 && !w->busy && w->r.state == 1 + reverse_phase &&
        w->model->active == w);
  unchanged(w, before);
  same_live(live);
  et_i64_tensor_borrow *positions = NULL;
  et_i64_tensor_error error;
  OK(et_i64_tensor_borrow_begin_v1(w->positions, &positions, &error));
  OK(et_i64_tensor_borrow_end_v1(&positions, &error));
  w->keep[0] = 1;
}

static void report_tensor(const et_f32_tensor *tensor) {
  uint32_t bits[1024];
  size_t count = 0;
  et_f32_tensor_error error;
  OK(et_f32_tensor_element_count_v1(tensor, &count, &error));
  CHECK(count <= sizeof(bits) / sizeof(*bits));
  OK(et_f32_tensor_copy_bits_to_v1(tensor, bits, count, &error));
  uint32_t magnitude = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t abs_bits = bits[i] & UINT32_C(0x7fffffff);
    CHECK(abs_bits < UINT32_C(0x7f800000));
    magnitude |= abs_bits;
  }
  CHECK(magnitude != 0);
  CHECK(fwrite(bits, sizeof(*bits), count, stdout) == count);
}

static int pa4_m3t_diagnostic(int argc, char **argv) {
  CHECK(argc == 2);
  select_pa4_wider_reference = !strcmp(argv[1], "reference");
  CHECK(select_pa4_wider_reference || !strcmp(argv[1], "a2"));

  initializer *s = et_m3t_private_initializer_create_v1(1729);
  CHECK(s);
  OK(et_m3t_private_initializer_seal_v1(s));
  owner *o = make_owner(s);
  check_attention_provider();
  const et_f32_test_live_counts_v1 baseline = live_counts();
  workspace *w = et_m3t_private_workspace_create_v1(o);
  input *in = et_m3t_private_input_create_v1();
  logits *upstream = et_m3t_private_logits_create_v1();
  logits *output = et_m3t_private_logits_create_v1();
  frame_copy *before = malloc(sizeof(*before));
  CHECK(w && in && upstream && output && before);

  const int64_t tokens[][2] = {{65, 66}, {255, 0}};
  for (size_t i = 0; i < sizeof(tokens) / sizeof(*tokens); ++i) {
    uint32_t dz[512];
    et_f32_tensor_error error;
    for (size_t j = 0; j < sizeof(dz) / sizeof(*dz); ++j)
      dz[j] = (j + i) % 3 == 0 ? UINT32_C(0xbf000000) : UINT32_C(0x3e800000);
    OK(et_f32_tensor_copy_bits_from_v1(upstream->tensor, dz, 512, &error));
    OK(et_m3t_private_input_copy_v1(in, tokens[i][0], tokens[i][1]));
    OK(et_m3t_private_workspace_begin_v1(w, in, 1));
    forward_before_attention(w);
    reject_invalid_attention_mask(w, 0, before);
    OK(et_m3t_private_attention_v1(w, 0));
    forward_after_attention(w);
    OK(et_m3t_private_workspace_copy_logits_v1(w, output));
    OK(et_m3t_private_workspace_vjp_begin_v1(w, upstream));
    reverse_before_attention(w);
    reject_invalid_attention_mask(w, 1, before);
    OK(et_m3t_private_attention_v1(w, 1));
    reverse_after_attention(w);
    OK(et_m3t_private_workspace_check_primals_v1(w, 1));

    report_tensor(output->tensor);
    report_tensor(w->slots[AH]);
    for (int slot = DQH; slot <= DVH; ++slot) report_tensor(w->slots[slot]);
    for (int64_t parameter = 0; parameter < 14; ++parameter)
      report_tensor(et_m3t_private_workspace_gradient_v1(w, parameter));
    OK(et_m3t_private_workspace_reset_v1(w));
    CHECK(!o->active && !et_m3t_private_workspace_gradient_v1(w, 0));
  }

  free(before);
  OK(et_m3t_private_workspace_release_v1(w));
  OK(et_m3t_private_workspace_release_v1(w));
  CHECK(et_m3t_private_workspace_begin_v1(w, in, 1) == ET_M3T_INVALID_STATE);
  OK(et_m3t_private_input_release_v1(in));
  OK(et_m3t_private_logits_release_v1(upstream));
  OK(et_m3t_private_logits_release_v1(output));
  same_live(baseline);
  CHECK(fflush(stdout) == 0);
  return 0;
}
