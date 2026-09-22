/* Development transport only: no public Eshkol facade or production symbol. */
#include "eshkol_transformer/g3s_sampling_abi.h"
#include <stdint.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
      ? et_g3s_kernel_provider_v1() : NULL;
}

int64_t et_g3s_test_provider_transport_v1(void) {
  const uint64_t row[] = {1,256}, rng_shape[] = {4}, token_shape[] = {1};
  float logits[256] = {0}, temperature = 1.0f, top_p = 1.0f;
  int64_t state[] = {1,0,-1,-1}, successor[4] = {0}, token = -1, top_k = 1;
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  et_kernel_request_v1 request = {sizeof(request), "g3s.greedy.forward", "f32", "cpu", 2, row, 1, {0}};
  et_kernel_tensor_view_v1 inputs[] = {
    {sizeof(inputs[0]), logits, sizeof(logits), "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 2, row},
    {sizeof(inputs[0]), state, sizeof(state), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 1, rng_shape},
    {sizeof(inputs[0]), &top_k, sizeof(top_k), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 0, NULL},
    {sizeof(inputs[0]), &top_p, sizeof(top_p), "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 0, NULL},
    {sizeof(inputs[0]), state, sizeof(state), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 1, rng_shape}
  };
  et_kernel_tensor_view_v1 outputs[] = {
    {sizeof(outputs[0]), &token, sizeof(token), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 1, token_shape},
    {sizeof(outputs[0]), successor, sizeof(successor), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 1, rng_shape}
  };
  et_kernel_call_v1 call = {sizeof(call), "g3s.greedy", &request,
      2,sizeof(inputs[0]),2*sizeof(inputs[0]),inputs,
      2,sizeof(outputs[0]),sizeof(outputs),outputs};
  if (et_kernel_runtime_discover(resolve,NULL,&runtime,&error) != 0) return 0;
  logits[42] = 3.0f;
  int success = et_kernel_runtime_dispatch(runtime,&call,&error) == 0 &&
      token == 42 && memcmp(state,successor,sizeof(state)) == 0;
  request.operation = "g3s.categorical.forward";
  call.capability = "g3s.categorical";
  call.input_count = 5;
  call.input_bytes = sizeof(inputs);
  inputs[1] = (et_kernel_tensor_view_v1){sizeof(inputs[0]), &temperature,
      sizeof(temperature), "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 0, NULL};
  /* Both complete outputs survive a required exhausted draw. */
  success = success && et_kernel_runtime_dispatch(runtime,&call,&error) != 0 &&
      token == 42 && memcmp(state,successor,sizeof(state)) == 0;
  state[2] = 0; state[3] = 0;
  for (int i = 0; i < 4 && success; ++i) {
    success = et_kernel_runtime_dispatch(runtime,&call,&error) == 0 && token == 42 &&
        successor[0] == 1 && successor[1] == 0 && successor[2] == i+1 && successor[3] == 0;
    memcpy(state,successor,sizeof(state));
  }
  et_kernel_runtime_destroy(runtime);
  return success ? 1 : 0;
}
#ifdef ET_G3S_AOT_BRIDGE_TESTING
int main(void) { return et_g3s_test_provider_transport_v1() == 1 ? 0 : 1; }
#endif
