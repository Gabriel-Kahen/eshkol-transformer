#include "test_common.h"
#include "../../native/m3t_c4_a2_empty_stage_vjp_private.h"
#include "eshkol_transformer/a2_kv_cache.h"
#include <fenv.h>

static float staged_k[16], staged_v[16], upstream_k[16], upstream_v[16];
static float gradient_k[16], gradient_v[16];
static const uint64_t shape[4] = {1u, 2u, 4u, 2u};
static const uint64_t count_shape[1] = {1u};

static void views(et_kernel_tensor_view_v1 a[4]) {
  a[0] = tv(upstream_k, sizeof(upstream_k), "f32", 4u, shape);
  a[1] = tv(upstream_v, sizeof(upstream_v), "f32", 4u, shape);
  a[2] = tv(gradient_k, sizeof(gradient_k), "f32", 4u, shape);
  a[3] = tv(gradient_v, sizeof(gradient_v), "f32", 4u, shape);
}

static int backward(et_kernel_tensor_view_v1 a[4]) {
  return et_m3t_c4_a2_empty_stage_vjp_private_v1(
      &a[0], &a[1], &a[2], &a[3]);
}

/* The finite-difference objective uses the real A2 stage and immutable
 * transaction view, not an imitation copy or an attention-only shortcut. */
static double loss(void) {
  et_a2_kv_cache *cache = NULL;
  et_a2_kv_cache_transaction *txn = NULL;
  et_a2_kv_cache_transaction_view *borrow = NULL;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  int64_t count = 4;
  et_kernel_tensor_view_v1 counts =
      tv(&count, sizeof(count), "i64", 1u, count_shape);
  et_kernel_tensor_view_v1 sk =
      tv(staged_k, sizeof(staged_k), "f32", 4u, shape);
  et_kernel_tensor_view_v1 sv =
      tv(staged_v, sizeof(staged_v), "f32", 4u, shape);
  CHECK(et_a2_kv_cache_create_v1(1u, 1u, 2u, 4u, 2u, &cache, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache, 4u, &counts, &txn, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      txn, 0u, &sk, &sv, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
      txn, 0u, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_view_tensors_v1(
      borrow, &k, &v, &lengths, &keep, &error) == 0);
  CHECK(k->byte_length == sizeof(staged_k));
  CHECK(v->byte_length == sizeof(staged_v));
  CHECK(((const int64_t *)lengths->data)[0] == 4);
  CHECK(memcmp(keep->data, "\1\1\1\1", 4u) == 0);
  CHECK(memcmp(k->data, staged_k, sizeof(staged_k)) == 0);
  CHECK(memcmp(v->data, staged_v, sizeof(staged_v)) == 0);
  const float *read_k = k->data, *read_v = v->data;
  double sum = 0.0;
  for (size_t i = 0u; i < 16u; ++i)
    sum += (double)read_k[i] * upstream_k[i] +
           (double)read_v[i] * upstream_v[i];
  CHECK(et_a2_kv_cache_transaction_view_end_v1(&borrow, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&txn, &error) == 0);
  CHECK(et_a2_kv_cache_destroy_v1(&cache, &error) == 0);
  return sum;
}

static void positive(void) {
  for (size_t i = 0u; i < 16u; ++i) {
    staged_k[i] = (float)((int)(i * 7u % 17u) - 8) / 16.0f;
    staged_v[i] = (float)((int)(i * 5u % 13u) - 6) / 16.0f;
    upstream_k[i] = (float)((int)(i * 11u % 19u) - 9) / 32.0f;
    upstream_v[i] = (float)((int)(i * 3u % 11u) - 5) / 32.0f;
  }
  memset(gradient_k, 0xa5, sizeof(gradient_k));
  memset(gradient_v, 0x5a, sizeof(gradient_v));
  et_kernel_tensor_view_v1 a[4];
  views(a);
  CHECK(backward(a) == ET_M3T_C4_A2_EMPTY_STAGE_VJP_OK);
  CHECK(memcmp(gradient_k, upstream_k, sizeof(gradient_k)) == 0);
  CHECK(memcmp(gradient_v, upstream_v, sizeof(gradient_v)) == 0);
  const float step = 0.00390625f;
  float *source[2] = {staged_k, staged_v};
  const float *gradient[2] = {gradient_k, gradient_v};
  for (size_t field = 0u; field < 2u; ++field)
    for (size_t i = 0u; i < 16u; ++i) {
      const float original = source[field][i];
      source[field][i] = original + step;
      const double high = loss();
      source[field][i] = original - step;
      const double low = loss();
      source[field][i] = original;
      CHECK(fabs((high - low) / (2.0 * step) - gradient[field][i]) < 0.0001);
    }
}

static void reject_unchanged(et_kernel_tensor_view_v1 a[4], int expected) {
  unsigned char before_k[sizeof(gradient_k)], before_v[sizeof(gradient_v)];
  memcpy(before_k, gradient_k, sizeof(before_k));
  memcpy(before_v, gradient_v, sizeof(before_v));
  CHECK(backward(a) == expected);
  CHECK(memcmp(before_k, gradient_k, sizeof(before_k)) == 0);
  CHECK(memcmp(before_v, gradient_v, sizeof(before_v)) == 0);
}

static void negatives(void) {
  et_kernel_tensor_view_v1 a[4];
  views(a); a[0].rank = 3u;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_SHAPE);
  views(a); a[1].byte_length = 60u;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_SHAPE);
  views(a); uint64_t wrong[4] = {1u, 2u, 3u, 2u}; a[2].shape = wrong;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_SHAPE);
  views(a); a[0].dtype = "i64";
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_DTYPE);
  views(a); a[1].device = "gpu";
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_DEVICE);
  views(a); a[2].offset_bytes = 4u;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_LAYOUT);
  views(a); a[3].layout = 0u;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_LAYOUT);
  views(a); a[2].data = upstream_k;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_OVERLAP);
  views(a); a[3].data = gradient_k;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_OVERLAP);
  views(a); upstream_k[0] = NAN;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_VALUE);
  upstream_k[0] = 0.0f; upstream_v[0] = INFINITY;
  reject_unchanged(a, ET_M3T_C4_A2_EMPTY_STAGE_VJP_VALUE);
  upstream_v[0] = 0.0f;
  CHECK(et_m3t_c4_a2_empty_stage_vjp_private_v1(
      NULL, &a[1], &a[2], &a[3]) ==
      ET_M3T_C4_A2_EMPTY_STAGE_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  positive();
  negatives();
  printf("PASS: C4 A2 empty-stage VJP checks=%zu\n", checks);
  return 0;
}
