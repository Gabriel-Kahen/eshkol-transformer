#include <stdint.h>
#include <string.h>
#include "eshkol_transformer/g3t_transport.h"
#include "eshkol_transformer/kernel_abi.h"

_Static_assert(sizeof(et_g3r_committed_history_snapshot_v1) ==
                   17 * sizeof(int64_t), "history snapshot layout changed");

/* Test-only link interposition. Successful calls always reach the real G3-S
 * provider; the one armed fault returns before invocation. */
static void *reference_owner;
static uint32_t prefix_logits[256];
static int64_t prefix_kv[2][4], prefix_rng[4];
static int64_t sample_calls, compared_calls, mismatch, inject_fault;

int64_t et_g3r_linked_probe_arm_v1(void *replay, void *reference,
                                   int64_t fault) {
  if (!replay || !reference || (fault != 0 && fault != 1) ||
      et_g3t_test_cache_length_v1(replay) != 1 ||
      et_g3t_test_pin_count_v1(replay) != 0) return -1;
  for (int64_t i = 0; i < 256; ++i) {
    int64_t word = et_g3t_test_logit_bits_v1(replay, i);
    if (word < 0) return -1;
    prefix_logits[i] = (uint32_t)word;
  }
  for (int64_t which = 0; which < 2; ++which)
    for (int64_t i = 0; i < 4; ++i) {
      int64_t word = et_g3t_test_manual_cache_kv_word_v1(replay, which, i);
      if (word < 0) return -1;
      prefix_kv[which][i] = word;
    }
  for (int64_t i = 0; i < 4; ++i) {
    prefix_rng[i] = et_g3t_test_rng_word_v1(replay, i);
    if (prefix_rng[i] < 0 && i < 2) return -1;
  }
  reference_owner = reference;
  compared_calls = 0;
  mismatch = 0;
  inject_fault = fault;
  return 0;
}

int64_t et_g3r_linked_probe_result_v1(void) {
  return compared_calls == 1 && !mismatch ? 0 : -1;
}
int64_t et_g3r_linked_sample_calls_v1(void) { return sample_calls; }

int64_t et_g3r_linked_history_word_v1(void *generator, int64_t index) {
  et_g3r_committed_history_snapshot_v1 snapshot;
  if (index < 0 || index >= 17 ||
      et_g3t_private_committed_history_snapshot_v1(generator, &snapshot))
    return -1;
  int64_t words[17];
  memcpy(words, &snapshot, sizeof(words));
  return words[index];
}

int32_t __real_et_kernel_runtime_dispatch(const et_kernel_runtime *,
                                           const et_kernel_call_v1 *,
                                           et_kernel_error *);
int32_t __wrap_et_kernel_runtime_dispatch(const et_kernel_runtime *runtime,
                                          const et_kernel_call_v1 *call,
                                          et_kernel_error *error) {
  if (call && call->capability &&
      strcmp(call->capability, "g3s.categorical") == 0) {
    ++sample_calls;
    if (reference_owner) {
      int bad = !call->inputs || call->input_count != 5 ||
                call->input_stride != sizeof(et_kernel_tensor_view_v1) ||
                et_g3t_test_cache_length_v1(reference_owner) != 1 ||
                et_g3t_test_pin_count_v1(reference_owner) <= 0;
      const et_kernel_tensor_view_v1 *inputs = call->inputs;
      if (!bad) {
        if (inputs[0].byte_length != sizeof(prefix_logits) ||
            !inputs[0].data || inputs[4].byte_length != sizeof(prefix_rng) ||
            !inputs[4].data ||
            memcmp(inputs[0].data, prefix_logits, sizeof(prefix_logits)) ||
            memcmp(inputs[4].data, prefix_rng, sizeof(prefix_rng))) bad = 1;
        for (int64_t which = 0; which < 2; ++which)
          for (int64_t i = 0; i < 4; ++i)
            if (et_g3t_test_manual_cache_kv_word_v1(
                    reference_owner, which, i) != prefix_kv[which][i])
              bad = 1;
      }
      mismatch |= bad;
      ++compared_calls;
      reference_owner = NULL;
      if (bad || inject_fault) {
        if (error) {
          memset(error, 0, sizeof(*error));
          error->category = ET_KERNEL_ERROR_INTERNAL;
          error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
          strcpy(error->operation, "g3r-linked-probe");
          strcpy(error->message, "armed sampler boundary");
        }
        inject_fault = 0;
        return ET_KERNEL_ERROR_INTERNAL;
      }
    }
  }
  return __real_et_kernel_runtime_dispatch(runtime, call, error);
}
