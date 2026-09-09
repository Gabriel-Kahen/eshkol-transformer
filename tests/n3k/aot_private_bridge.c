#include "test_common.h"

/* Test-only fixed buffers: no public Eshkol tensor/model transport. */
static void dispatch(et_kernel_runtime *runtime, const char *capability,
                     const char *operation, size_t rank, const uint64_t *shape,
                     et_kernel_tensor_view_v1 *inputs, size_t ni,
                     et_kernel_tensor_view_v1 *outputs, size_t no) {
  et_kernel_request_v1 request = rq(operation, rank, shape);
  et_kernel_call_v1 call = kc(capability, &request, inputs, ni, outputs, no);
  run(runtime, &call);
}

int64_t et_n3k_test_provider_transport_v1(void) {
  et_kernel_runtime *runtime = runtime_new();
  const uint64_t matrix_shape[] = {2, 4}, state_shape[] = {4};
  const uint64_t activation_shape[] = {1, 2, 4}, id_shape[] = {1, 2};
  float weight[8], y[8], dweight[8];
  int64_t state[] = {1, 1729, 0, 0}, next[4];
  et_kernel_tensor_view_v1 ii[] = {tv(state, sizeof(state), "i64", 1, state_shape)};
  et_kernel_tensor_view_v1 io[] = {
      tv(weight, sizeof(weight), "f32", 2, matrix_shape),
      tv(next, sizeof(next), "i64", 1, state_shape)};
  dispatch(runtime, "n3k.matrix-init", "n3k.matrix-init.uniform", 2,
           matrix_shape, ii, 1, io, 2);
  CHECK(next[0] == 1 && next[1] == 1729 && next[2] == 2 && next[3] == 0);
  for (size_t i = 0; i < COUNT(weight); i++)
    CHECK(weight[i] >= -0.03125f && weight[i] < 0.03125f);

  {
    const uint64_t row[] = {1, 2, 2, 4};
    int64_t ids[] = {1, 1};
    float upstream[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float expected[] = {0, 0, 0, 0, 6, 8, 10, 12};
    et_kernel_tensor_view_v1 inputs[] = {
        tv(ids, sizeof(ids), "i64", 2, id_shape),
        tv(weight, sizeof(weight), "f32", 2, matrix_shape)};
    et_kernel_tensor_view_v1 outputs[] = {
        tv(y, sizeof(y), "f32", 3, activation_shape)};
    dispatch(runtime, "n3k.embedding-forward", "n3k.embedding.forward",
             4, row, inputs, 2, outputs, 1);
    CHECK(memcmp(y, weight + 4, 4 * sizeof(float)) == 0);
    CHECK(memcmp(y + 4, weight + 4, 4 * sizeof(float)) == 0);
    inputs[1] = tv(upstream, sizeof(upstream), "f32", 3, activation_shape);
    outputs[0] = tv(dweight, sizeof(dweight), "f32", 2, matrix_shape);
    dispatch(runtime, "n3k.embedding-backward", "n3k.embedding.backward",
             4, row, inputs, 2, outputs, 1);
    CHECK(memcmp(dweight, expected, sizeof(expected)) == 0);
  }
  {
    const uint64_t row[] = {1, 2, 2, 2};
    float token[] = {0, 1, 10, 11, 100, 101, 110, 111}, head[8], restored[8];
    const float expected[] = {0, 1, 100, 101, 10, 11, 110, 111};
    const char *to_head[] = {"n3k.heads.split.forward", "n3k.heads.merge.backward"};
    const char *to_token[] = {"n3k.heads.merge.forward", "n3k.heads.split.backward"};
    et_kernel_tensor_view_v1 ti[] = {tv(token, sizeof(token), "f32", 3, activation_shape)};
    et_kernel_tensor_view_v1 ho[] = {tv(head, sizeof(head), "f32", 4, row)};
    et_kernel_tensor_view_v1 ro[] = {tv(restored, sizeof(restored), "f32", 3, activation_shape)};
    for (size_t i = 0; i < COUNT(to_head); i++) {
      dispatch(runtime, "n3k.head-layout", to_head[i], 4, row, ti, 1, ho, 1);
      CHECK(memcmp(head, expected, sizeof(expected)) == 0);
      dispatch(runtime, "n3k.head-layout", to_token[i], 4, row, ho, 1, ro, 1);
      CHECK(memcmp(restored, token, sizeof(token)) == 0);
    }
  }
  {
    const uint64_t row[] = {1, 2, 4, 4}, ws[] = {4, 4};
    float x[] = {1, 2, 3, 4, 5, 6, 7, 8}, w[16] = {0}, dy[8], dx[8], dw[16];
    for (size_t i = 0; i < 4; i++) w[i * 4 + i] = 2.0f;
    for (size_t i = 0; i < 8; i++) dy[i] = 1.0f;
    et_kernel_tensor_view_v1 inputs[] = {
        tv(x, sizeof(x), "f32", 3, activation_shape),
        tv(w, sizeof(w), "f32", 2, ws),
        tv(dy, sizeof(dy), "f32", 3, activation_shape)};
    et_kernel_tensor_view_v1 outputs[] = {tv(y, sizeof(y), "f32", 3, activation_shape)};
    dispatch(runtime, "n3k.linear", "n3k.linear.forward-no-bias", 4, row, inputs, 2, outputs, 1);
    for (size_t i = 0; i < 8; i++) CHECK(y[i] == 2.0f * x[i]);
    et_kernel_tensor_view_v1 gradients[] = {
        tv(dx, sizeof(dx), "f32", 3, activation_shape), tv(dw, sizeof(dw), "f32", 2, ws)};
    dispatch(runtime, "n3k.linear", "n3k.linear.backward-no-bias", 4, row, inputs, 3, gradients, 2);
    for (size_t i = 0; i < 8; i++) CHECK(dx[i] == 2.0f);
    for (size_t i = 0; i < 16; i++) CHECK(dw[i] == x[i % 4] + x[i % 4 + 4]);
  }
  {
    const uint64_t shape[] = {1, 2, 8};
    float zero[16] = {0}, upstream[16], out[16];
    for (size_t i = 0; i < 16; i++) upstream[i] = 2.0f;
    et_kernel_tensor_view_v1 inputs[] = {
        tv(zero, sizeof(zero), "f32", 3, shape), tv(upstream, sizeof(upstream), "f32", 3, shape)};
    et_kernel_tensor_view_v1 outputs[] = {tv(out, sizeof(out), "f32", 3, shape)};
    dispatch(runtime, "n3k.gelu", "n3k.gelu.forward", 3, shape, inputs, 1, outputs, 1);
    for (size_t i = 0; i < 16; i++) CHECK(fbits(out[i]) == 0);
    dispatch(runtime, "n3k.gelu", "n3k.gelu.backward", 3, shape, inputs, 2, outputs, 1);
    for (size_t i = 0; i < 16; i++) CHECK(out[i] == 1.0f);
  }
  {
    float a[8], b[8], c[8], out[8], da[8], db[8], dc[8];
    for (size_t i = 0; i < 8; i++) { a[i] = 1; b[i] = 2; c[i] = 4; }
    et_kernel_tensor_view_v1 inputs[] = {
        tv(a, sizeof(a), "f32", 3, activation_shape), tv(b, sizeof(b), "f32", 3, activation_shape),
        tv(c, sizeof(c), "f32", 3, activation_shape)};
    et_kernel_tensor_view_v1 outputs[] = {tv(out, sizeof(out), "f32", 3, activation_shape)};
    et_kernel_tensor_view_v1 gradients[] = {
        tv(da, sizeof(da), "f32", 3, activation_shape), tv(db, sizeof(db), "f32", 3, activation_shape),
        tv(dc, sizeof(dc), "f32", 3, activation_shape)};
    const char *caps[] = {"n3k.residual", "n3k.activation-sum", "n3k.activation-sum"};
    const char *fw[] = {"n3k.residual.forward", "n3k.sum2.forward", "n3k.sum3.forward"};
    const char *bw[] = {"n3k.residual.backward", "n3k.sum2.backward", "n3k.sum3.backward"};
    for (size_t op = 0; op < 3; op++) {
      const size_t edges = op == 2 ? 3 : 2;
      dispatch(runtime, caps[op], fw[op], 3, activation_shape, inputs, edges, outputs, 1);
      for (size_t i = 0; i < 8; i++) CHECK(out[i] == (edges == 3 ? 7.0f : 3.0f));
      dispatch(runtime, caps[op], bw[op], 3, activation_shape, outputs, 1, gradients, edges);
      CHECK(memcmp(da, out, sizeof(out)) == 0 && memcmp(db, out, sizeof(out)) == 0);
      if (edges == 3) CHECK(memcmp(dc, out, sizeof(out)) == 0);
    }
  }
  {
    const uint64_t shape[] = {256, 4};
    float a[1024], b[1024], out[1024], da[1024], db[1024];
    for (size_t i = 0; i < 1024; i++) { a[i] = 1; b[i] = 2; }
    et_kernel_tensor_view_v1 inputs[] = {tv(a, sizeof(a), "f32", 2, shape), tv(b, sizeof(b), "f32", 2, shape)};
    et_kernel_tensor_view_v1 outputs[] = {tv(out, sizeof(out), "f32", 2, shape)};
    et_kernel_tensor_view_v1 gradients[] = {tv(da, sizeof(da), "f32", 2, shape), tv(db, sizeof(db), "f32", 2, shape)};
    dispatch(runtime, "n3k.tied-sum", "n3k.sum2.forward", 2, shape, inputs, 2, outputs, 1);
    for (size_t i = 0; i < 1024; i++) CHECK(out[i] == 3);
    dispatch(runtime, "n3k.tied-sum", "n3k.sum2.backward", 2, shape, outputs, 1, gradients, 2);
    CHECK(memcmp(da, out, sizeof(out)) == 0 && memcmp(db, out, sizeof(out)) == 0);
  }
  et_kernel_runtime_destroy(runtime);
  return 1;
}

#if defined(ET_N3K_AOT_BRIDGE_TESTING)
int main(void) { return et_n3k_test_provider_transport_v1() == 1 ? 0 : 1; }
#endif
