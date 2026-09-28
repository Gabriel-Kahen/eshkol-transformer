#include <eshkol/eshkol.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void et_e1b_public_tr3_metrics_ref_v1(void *, void *, void *);

static int64_t next_code;
static int calls;

eshkol_tagged_value_t *et_e1b_box_value_v1(void *box) {
  return (eshkol_tagged_value_t *)box;
}

void et_e1b_ensure_private_initialized_v1(void) {}

eshkol_tagged_value_t et_e1b_private_tr3_metrics_ref_cabi_v1(
    eshkol_tagged_value_t metrics, eshkol_tagged_value_t key) {
  (void)metrics;
  (void)key;
  ++calls;
  return eshkol_make_int64(next_code, true);
}

static void check(int condition) {
  if (!condition) abort();
}

int main(void) {
  const uint32_t bits[] = {UINT32_C(0x00000001), UINT32_C(0x80000000),
                           UINT32_C(0x3eaaaaab), UINT32_C(0x7f7fffff)};
  eshkol_tagged_value_t metrics = eshkol_make_int64(1, true);
  eshkol_tagged_value_t key = eshkol_make_int64(2, true);
  eshkol_tagged_value_t output = eshkol_make_int64(-1, true);
  check(eshkol_runtime_has_f32_scalar_v1() == 1);
  for (size_t i = 0; i < sizeof(bits) / sizeof(bits[0]); ++i) {
    uint32_t returned = 0;
    next_code = -INT64_C(1) - (int64_t)bits[i];
    et_e1b_public_tr3_metrics_ref_v1(&metrics, &key, &output);
    check(eshkol_value_is_f32_v1(&output) == 1);
    check(eshkol_value_f32_to_bits_v1(&output, &returned) ==
          ESHKOL_VALUE_F32_OK);
    check(returned == bits[i]);
  }
  next_code = INT64_MAX;
  et_e1b_public_tr3_metrics_ref_v1(&metrics, &key, &output);
  check(output.type == ESHKOL_VALUE_INT64);
  check(output.flags == ESHKOL_VALUE_EXACT_FLAG);
  check(output.data.int_val == INT64_MAX);
  check(calls == 5);
  puts("TR3-PUBLIC-METRICS-BRIDGE-PASS");
  return 0;
}
