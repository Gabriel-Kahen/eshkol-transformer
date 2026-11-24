#include "eshkol_transformer/mod5_moe_abi.h"
#include "mod5_moe_eshkol_bridge.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 MoE native test requires x86-64"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Trap any heap allocation during provider/direct/bridge dispatch. */
static int dispatching = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!dispatching);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!dispatching);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!dispatching);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!dispatching);
  __real_free(ptr);
}

static const et_kernel_provider_v1 *resolve_moe(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_moe_kernel_provider_v1()
             : NULL;
}

static et_kernel_runtime *discover_moe(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_moe, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t rank, const uint64_t *shape) {
  size_t bytes = strcmp(dtype, "i64") == 0 ? sizeof(int64_t) : sizeof(float);
  for (size_t d = 0; d < rank; ++d)
    bytes *= (size_t)shape[d];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static int32_t dispatch_moe_forward_k1(const et_kernel_runtime *runtime,
                                       size_t n, size_t t, size_t d, size_t f,
                                       size_t e, size_t k, void *x,
                                       void *w_gate, void *w_gate_exp,
                                       void *w_up_exp, void *w_down_exp,
                                       float *alpha_aux, void *y,
                                       void *aux_loss, void *expert_counts,
                                       void *active_evals,
                                       et_kernel_error *error) {
  const uint64_t shape[] = {n, t, d, f, e, k};
  const uint64_t x_shape[] = {n, t, d};
  const uint64_t wg_shape[] = {e, d};
  const uint64_t wgu_shape[] = {e, f, d};
  const uint64_t wd_shape[] = {e, d, f};
  const uint64_t cnt_shape[] = {e};
  const et_kernel_tensor_view_v1 in[] = {
      make_view(x, "f32", 3u, x_shape),
      make_view(w_gate, "f32", 2u, wg_shape),
      make_view(w_gate_exp, "f32", 3u, wgu_shape),
      make_view(w_up_exp, "f32", 3u, wgu_shape),
      make_view(w_down_exp, "f32", 3u, wd_shape),
      make_view(alpha_aux, "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 out[] = {
      make_view(y, "f32", 3u, x_shape),
      make_view(aux_loss, "f32", 0u, NULL),
      make_view(expert_counts, "i64", 1u, cnt_shape),
      make_view(active_evals, "i64", 0u, NULL)};
  const et_kernel_request_v1 request = {sizeof(request), "moe.forward",
                                        "f32",           "cpu",
                                        6u,              shape,
                                        1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.moe", &request, 6u, stride, 6u * stride, in,
      4u,           stride,       4u * stride, out};
  return et_kernel_runtime_dispatch(runtime, &call, error);
}

static int32_t dispatch_moe_backward_k1(const et_kernel_runtime *runtime,
                                        size_t n, size_t t, size_t d, size_t f,
                                        size_t e, size_t k, void *x,
                                        void *w_gate, void *w_gate_exp,
                                        void *w_up_exp, void *w_down_exp,
                                        float *alpha_aux, void *dy, void *dx,
                                        void *dw_gate, void *dw_gate_exp,
                                        void *dw_up_exp, void *dw_down_exp,
                                        et_kernel_error *error) {
  const uint64_t shape[] = {n, t, d, f, e, k};
  const uint64_t x_shape[] = {n, t, d};
  const uint64_t wg_shape[] = {e, d};
  const uint64_t wgu_shape[] = {e, f, d};
  const uint64_t wd_shape[] = {e, d, f};
  const et_kernel_tensor_view_v1 in[] = {
      make_view(x, "f32", 3u, x_shape),
      make_view(w_gate, "f32", 2u, wg_shape),
      make_view(w_gate_exp, "f32", 3u, wgu_shape),
      make_view(w_up_exp, "f32", 3u, wgu_shape),
      make_view(w_down_exp, "f32", 3u, wd_shape),
      make_view(alpha_aux, "f32", 0u, NULL),
      make_view(dy, "f32", 3u, x_shape)};
  et_kernel_tensor_view_v1 out[] = {
      make_view(dx, "f32", 3u, x_shape),
      make_view(dw_gate, "f32", 2u, wg_shape),
      make_view(dw_gate_exp, "f32", 3u, wgu_shape),
      make_view(dw_up_exp, "f32", 3u, wgu_shape),
      make_view(dw_down_exp, "f32", 3u, wd_shape)};
  const et_kernel_request_v1 request = {sizeof(request), "moe.backward",
                                        "f32",           "cpu",
                                        6u,              shape,
                                        1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.moe", &request, 7u, stride, 7u * stride, in,
      5u,           stride,       5u * stride, out};
  return et_kernel_runtime_dispatch(runtime, &call, error);
}

static void fill_deterministic_f32(float *buf, size_t count, int salt,
                                   float scale) {
  for (size_t i = 0; i < count; ++i) {
    const int v = (int)((37u * (unsigned)salt + 19u * (unsigned)i +
                         7u * (unsigned)(i / 3u)) %
                        43u) -
                  21;
    buf[i] = (float)v * scale;
  }
}

/* Fill router weights with well-separated logits so FD stays off tie walls. */
static void fill_router_inputs(float *x, size_t total_tokens, size_t model_dim,
                               float *w_gate, size_t num_experts, int salt) {
  fill_deterministic_f32(x, total_tokens * model_dim, salt, 0.0625f);
  fill_deterministic_f32(w_gate, num_experts * model_dim, salt + 17, 0.0625f);
  for (size_t s = 0; s < total_tokens; ++s) {
    x[s * model_dim] = 0.75f + 0.0625f * (float)(s % 5u);
    if (model_dim > 1u)
      x[s * model_dim + 1u] = ((s & 1u) ? -0.5f : 0.5f) + 0.03125f * (float)s;
  }
  for (size_t e = 0; e < num_experts; ++e) {
    w_gate[e * model_dim] = 0.25f * (float)((int)e - (int)(num_experts / 2u));
    if (model_dim > 1u)
      w_gate[e * model_dim + 1u] =
          ((e & 1u) ? 0.375f : -0.375f) + 0.0625f * (float)e;
  }
}

typedef struct moe_case {
  size_t n, t, d, f, e, k;
  float alpha_aux;
  int salt;
  int check_fd;
} moe_case;

static const moe_case k_cases[] = {
    {1, 1, 2, 2, 2, 1, 0.0625f, 1, 1},
    {1, 4, 4, 8, 4, 2, 0.125f, 2, 1},
    {2, 6, 8, 12, 4, 1, 0.25f, 3, 1},
    {1, 8, 8, 16, 6, 3, 0.0625f, 4, 1},
    {2, 8, 12, 16, 8, 2, 0.125f, 5, 1},
    {1, 4, 6, 8, 4, 4, 0.0f, 6, 1},
    {2, 32, 32, 64, 8, 4, 0.0625f, 7, 0}, /* Max boundary shape check */
};

static double dot_f64(const float *a, const float *b, size_t count) {
  double acc = 0.0;
  for (size_t i = 0; i < count; ++i)
    acc += (double)a[i] * (double)b[i];
  return acc;
}

static void check_fd_tensor(const moe_case *mc, float *target_buf,
                            const float *analytic_grad, size_t count,
                            float *x, float *w_gate, float *w_ge, float *w_ue,
                            float *w_de, const float *dy,
                            const int64_t *base_counts, float *y_tmp,
                            int64_t *cnt_plus, int64_t *cnt_minus,
                            double *max_fd_err, size_t *fd_checks) {
  const float eps = 0.001953125f; /* 2^-9 */
  const size_t x_count = mc->n * mc->t * mc->d;
  const size_t max_samples = 16u;
  const size_t stride = count <= max_samples ? 1u : (count / max_samples);
  size_t checked_here = 0u;
  et_kernel_error error;

  for (size_t idx = 0; idx < count; idx += stride) {
    const float orig = target_buf[idx];
    float aux_plus = 0.0f, aux_minus = 0.0f;
    int64_t evals_plus = 0, evals_minus = 0;

    target_buf[idx] = orig + eps;
    dispatching = 1;
    CHECK(et_mod5_moe_forward_f32(mc->n, mc->t, mc->d, mc->f, mc->e, mc->k, x,
                                  w_gate, w_ge, w_ue, w_de,
                                  (double)mc->alpha_aux, y_tmp, &aux_plus,
                                  cnt_plus, &evals_plus, &error) == 0);
    const double lp = dot_f64(dy, y_tmp, x_count) + (double)aux_plus;

    target_buf[idx] = orig - eps;
    CHECK(et_mod5_moe_forward_f32(mc->n, mc->t, mc->d, mc->f, mc->e, mc->k, x,
                                  w_gate, w_ge, w_ue, w_de,
                                  (double)mc->alpha_aux, y_tmp, &aux_minus,
                                  cnt_minus, &evals_minus, &error) == 0);
    dispatching = 0;
    const double lm = dot_f64(dy, y_tmp, x_count) + (double)aux_minus;
    target_buf[idx] = orig;

    /* Skip only if perturbation crossed a discrete top-K tie boundary. */
    if (memcmp(cnt_plus, base_counts, mc->e * sizeof(int64_t)) != 0 ||
        memcmp(cnt_minus, base_counts, mc->e * sizeof(int64_t)) != 0)
      continue;

    const double num = (lp - lm) / (2.0 * (double)eps);
    const double ana = (double)analytic_grad[idx];
    const double diff = fabs(ana - num);
    if (diff > *max_fd_err)
      *max_fd_err = diff;
    CHECK(diff <= 2e-4 + 1e-3 * fabs(num));
    (*fd_checks)++;
    checked_here++;
  }
  CHECK(checked_here > 0u);
}

static void run_case(const et_kernel_runtime *runtime, const moe_case *mc,
                     double *max_fd_err, size_t *fd_checks) {
  const size_t x_cnt = mc->n * mc->t * mc->d;
  const size_t wg_cnt = mc->e * mc->d;
  const size_t exp_cnt = mc->e * mc->f * mc->d;

  float *x = (float *)calloc(x_cnt, sizeof(float));
  float *w_gate = (float *)calloc(wg_cnt, sizeof(float));
  float *w_ge = (float *)calloc(exp_cnt, sizeof(float));
  float *w_ue = (float *)calloc(exp_cnt, sizeof(float));
  float *w_de = (float *)calloc(exp_cnt, sizeof(float));
  float *dy = (float *)calloc(x_cnt, sizeof(float));

  float *y = (float *)calloc(x_cnt, sizeof(float));
  float *y_rep = (float *)calloc(x_cnt, sizeof(float));
  float *y_dir = (float *)calloc(x_cnt, sizeof(float));
  float *y_tmp = (float *)calloc(x_cnt, sizeof(float));

  float *dx = (float *)calloc(x_cnt, sizeof(float));
  float *dw_gate = (float *)calloc(wg_cnt, sizeof(float));
  float *dw_ge = (float *)calloc(exp_cnt, sizeof(float));
  float *dw_ue = (float *)calloc(exp_cnt, sizeof(float));
  float *dw_de = (float *)calloc(exp_cnt, sizeof(float));

  float *dx_dir = (float *)calloc(x_cnt, sizeof(float));
  float *dw_gate_dir = (float *)calloc(wg_cnt, sizeof(float));
  float *dw_ge_dir = (float *)calloc(exp_cnt, sizeof(float));
  float *dw_ue_dir = (float *)calloc(exp_cnt, sizeof(float));
  float *dw_de_dir = (float *)calloc(exp_cnt, sizeof(float));

  int64_t *counts = (int64_t *)calloc(mc->e, sizeof(int64_t));
  int64_t *counts_rep = (int64_t *)calloc(mc->e, sizeof(int64_t));
  int64_t *counts_dir = (int64_t *)calloc(mc->e, sizeof(int64_t));
  int64_t *cnt_plus = (int64_t *)calloc(mc->e, sizeof(int64_t));
  int64_t *cnt_minus = (int64_t *)calloc(mc->e, sizeof(int64_t));

  CHECK(x && w_gate && w_ge && w_ue && w_de && dy && y && y_rep && y_dir &&
        y_tmp && dx && dw_gate && dw_ge && dw_ue && dw_de && dx_dir &&
        dw_gate_dir && dw_ge_dir && dw_ue_dir && dw_de_dir && counts &&
        counts_rep && counts_dir && cnt_plus && cnt_minus);

  fill_router_inputs(x, mc->n * mc->t, mc->d, w_gate, mc->e, mc->salt);
  fill_deterministic_f32(w_ge, exp_cnt, mc->salt + 31, 0.03125f);
  fill_deterministic_f32(w_ue, exp_cnt, mc->salt + 53, 0.03125f);
  fill_deterministic_f32(w_de, exp_cnt, mc->salt + 79, 0.03125f);
  fill_deterministic_f32(dy, x_cnt, mc->salt + 101, 0.0625f);

  float alpha = mc->alpha_aux;
  float aux = 0.0f, aux_rep = 0.0f, aux_dir = 0.0f;
  int64_t evals = 0, evals_rep = 0, evals_dir = 0;
  et_kernel_error error;

  dispatching = 1;
  CHECK(dispatch_moe_forward_k1(runtime, mc->n, mc->t, mc->d, mc->f, mc->e,
                                mc->k, x, w_gate, w_ge, w_ue, w_de, &alpha, y,
                                &aux, counts, &evals, &error) == 0);
  CHECK(dispatch_moe_forward_k1(runtime, mc->n, mc->t, mc->d, mc->f, mc->e,
                                mc->k, x, w_gate, w_ge, w_ue, w_de, &alpha,
                                y_rep, &aux_rep, counts_rep, &evals_rep,
                                &error) == 0);
  CHECK(et_mod5_moe_forward_f32(mc->n, mc->t, mc->d, mc->f, mc->e, mc->k, x,
                                w_gate, w_ge, w_ue, w_de, (double)alpha, y_dir,
                                &aux_dir, counts_dir, &evals_dir, &error) == 0);

  CHECK(dispatch_moe_backward_k1(runtime, mc->n, mc->t, mc->d, mc->f, mc->e,
                                 mc->k, x, w_gate, w_ge, w_ue, w_de, &alpha, dy,
                                 dx, dw_gate, dw_ge, dw_ue, dw_de, &error) == 0);
  CHECK(et_mod5_moe_backward_f32(mc->n, mc->t, mc->d, mc->f, mc->e, mc->k, x,
                                 w_gate, w_ge, w_ue, w_de, (double)alpha, dy,
                                 dx_dir, dw_gate_dir, dw_ge_dir, dw_ue_dir,
                                 dw_de_dir, &error) == 0);
  dispatching = 0;

  /* Bitwise determinism & K1-vs-direct parity. */
  CHECK(memcmp(y, y_rep, x_cnt * sizeof(float)) == 0);
  CHECK(memcmp(y, y_dir, x_cnt * sizeof(float)) == 0);
  CHECK(memcmp(&aux, &aux_rep, sizeof(float)) == 0);
  CHECK(memcmp(&aux, &aux_dir, sizeof(float)) == 0);
  CHECK(memcmp(counts, counts_rep, mc->e * sizeof(int64_t)) == 0);
  CHECK(memcmp(counts, counts_dir, mc->e * sizeof(int64_t)) == 0);
  CHECK(evals == evals_rep && evals == evals_dir);
  CHECK(memcmp(dx, dx_dir, x_cnt * sizeof(float)) == 0);
  CHECK(memcmp(dw_gate, dw_gate_dir, wg_cnt * sizeof(float)) == 0);
  CHECK(memcmp(dw_ge, dw_ge_dir, exp_cnt * sizeof(float)) == 0);
  CHECK(memcmp(dw_ue, dw_ue_dir, exp_cnt * sizeof(float)) == 0);
  CHECK(memcmp(dw_de, dw_de_dir, exp_cnt * sizeof(float)) == 0);

  /* Sparse dispatch telemetry verification. */
  const int64_t expected_evals = (int64_t)(mc->n * mc->t * mc->k);
  const int64_t dense_evals = (int64_t)(mc->n * mc->t * mc->e);
  CHECK(evals == expected_evals);
  if (mc->k < mc->e)
    CHECK(evals < dense_evals);
  int64_t sum_counts = 0;
  for (size_t e = 0; e < mc->e; ++e) {
    CHECK(counts[e] >= 0 && counts[e] <= (int64_t)(mc->n * mc->t));
    sum_counts += counts[e];
  }
  CHECK(sum_counts == expected_evals);
  if (mc->alpha_aux == 0.0f)
    CHECK(aux == 0.0f);
  else
    CHECK(aux > 0.0f && isfinite(aux));

  /* Central finite-difference checks on all 5 analytic gradients. */
  if (mc->check_fd) {
    check_fd_tensor(mc, x, dx, x_cnt, x, w_gate, w_ge, w_ue, w_de, dy, counts,
                    y_tmp, cnt_plus, cnt_minus, max_fd_err, fd_checks);
    check_fd_tensor(mc, w_gate, dw_gate, wg_cnt, x, w_gate, w_ge, w_ue, w_de,
                    dy, counts, y_tmp, cnt_plus, cnt_minus, max_fd_err,
                    fd_checks);
    check_fd_tensor(mc, w_ge, dw_ge, exp_cnt, x, w_gate, w_ge, w_ue, w_de, dy,
                    counts, y_tmp, cnt_plus, cnt_minus, max_fd_err, fd_checks);
    check_fd_tensor(mc, w_ue, dw_ue, exp_cnt, x, w_gate, w_ge, w_ue, w_de, dy,
                    counts, y_tmp, cnt_plus, cnt_minus, max_fd_err, fd_checks);
    check_fd_tensor(mc, w_de, dw_de, exp_cnt, x, w_gate, w_ge, w_ue, w_de, dy,
                    counts, y_tmp, cnt_plus, cnt_minus, max_fd_err, fd_checks);
  }

  free(x);
  free(w_gate);
  free(w_ge);
  free(w_ue);
  free(w_de);
  free(dy);
  free(y);
  free(y_rep);
  free(y_dir);
  free(y_tmp);
  free(dx);
  free(dw_gate);
  free(dw_ge);
  free(dw_ue);
  free(dw_de);
  free(dx_dir);
  free(dw_gate_dir);
  free(dw_ge_dir);
  free(dw_ue_dir);
  free(dw_de_dir);
  free(counts);
  free(counts_rep);
  free(counts_dir);
  free(cnt_plus);
  free(cnt_minus);
}

static void check_sparse_skip_tie_breaking_and_router_decomp(
    const et_kernel_runtime *runtime) {
  et_kernel_error error;

  /* A. Prove unselected experts are truly skipped in forward & backward. */
  {
    const size_t n = 1, t = 4, d = 4, f = 6, e = 4, k = 2;
    const size_t exp_stride = f * d;
    float x[16], w_gate[16], w_ge[96], w_ue[96], w_de[96], dy[16];
    float y1[16], y2[16], aux1 = 0.0f, aux2 = 0.0f;
    float dx1[16], dx2[16], dwg1[16], dwg2[16];
    float dwge1[96], dwge2[96], dwue1[96], dwue2[96], dwde1[96], dwde2[96];
    int64_t c1[4], c2[4], evals1 = 0, evals2 = 0;

    fill_deterministic_f32(x, 16u, 11, 0.0625f);
    for (size_t s = 0; s < t; ++s)
      x[s * d] = 1.0f;
    memset(w_gate, 0, sizeof(w_gate));
    /* Strongly suppress expert 3 so counts[3] == 0 across all tokens. */
    w_gate[0 * d] = 2.0f;
    w_gate[1 * d] = 1.5f;
    w_gate[2 * d] = 0.5f;
    w_gate[3 * d] = -8.0f;
    fill_deterministic_f32(w_ge, 96u, 13, 0.0625f);
    fill_deterministic_f32(w_ue, 96u, 17, 0.0625f);
    fill_deterministic_f32(w_de, 96u, 19, 0.0625f);
    fill_deterministic_f32(dy, 16u, 23, 0.0625f);

    float alpha = 0.125f;
    dispatching = 1;
    CHECK(dispatch_moe_forward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                  w_ue, w_de, &alpha, y1, &aux1, c1, &evals1,
                                  &error) == 0);
    CHECK(dispatch_moe_backward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                   w_ue, w_de, &alpha, dy, dx1, dwg1, dwge1,
                                   dwue1, dwde1, &error) == 0);
    CHECK(evals1 == 8 && evals1 < 16);
    CHECK(c1[0] == 4 && c1[1] == 4 && c1[2] == 0 && c1[3] == 0);

    /* Unselected experts 2 and 3 must have identically zero weight grads. */
    for (size_t idx = 2u * exp_stride; idx < 4u * exp_stride; ++idx) {
      CHECK(dwge1[idx] == 0.0f && dwue1[idx] == 0.0f && dwde1[idx] == 0.0f);
    }

    /* Mutate finite weights of unselected experts 2 and 3 drastically. */
    for (size_t idx = 2u * exp_stride; idx < 4u * exp_stride; ++idx) {
      w_ge[idx] = 12.5f;
      w_ue[idx] = -9.25f;
      w_de[idx] = 7.75f;
    }
    CHECK(dispatch_moe_forward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                  w_ue, w_de, &alpha, y2, &aux2, c2, &evals2,
                                  &error) == 0);
    CHECK(dispatch_moe_backward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                   w_ue, w_de, &alpha, dy, dx2, dwg2, dwge2,
                                   dwue2, dwde2, &error) == 0);
    CHECK(memcmp(y1, y2, sizeof(y1)) == 0);
    CHECK(memcmp(&aux1, &aux2, sizeof(float)) == 0);
    CHECK(memcmp(c1, c2, sizeof(c1)) == 0);
    CHECK(evals1 == evals2);
    CHECK(memcmp(dx1, dx2, sizeof(dx1)) == 0);
    CHECK(memcmp(dwg1, dwg2, sizeof(dwg1)) == 0);
    CHECK(memcmp(dwge1, dwge2, sizeof(dwge1)) == 0);
    CHECK(memcmp(dwue1, dwue2, sizeof(dwue1)) == 0);
    CHECK(memcmp(dwde1, dwde2, sizeof(dwde1)) == 0);

    /* NaN inside an unselected expert MUST STILL reject explicitly! */
    w_ge[3u * exp_stride + 2u] = NAN;
    CHECK(dispatch_moe_forward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                  w_ue, w_de, &alpha, y2, &aux2, c2, &evals2,
                                  &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(dispatch_moe_backward_k1(runtime, n, t, d, f, e, k, x, w_gate, w_ge,
                                   w_ue, w_de, &alpha, dy, dx2, dwg2, dwge2,
                                   dwue2, dwde2, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    dispatching = 0;
  }

  /* B. Deterministic lowest-expert-index tie-breaking (full & partial ties). */
  {
    const size_t n = 1, t = 3, d = 2, f = 2, e = 6, k = 3;
    float x[6] = {1.0f, 0.5f, 1.0f, -0.5f, 1.0f, 0.25f};
    float w_gate[12];
    float w_ge[24], w_ue[24], w_de[24], y[6], aux = 0.0f;
    int64_t counts[6], evals = 0;
    fill_deterministic_f32(w_ge, 24u, 41, 0.125f);
    fill_deterministic_f32(w_ue, 24u, 43, 0.125f);
    fill_deterministic_f32(w_de, 24u, 47, 0.125f);

    /* Full 6-way tie (W_gate = 0): must select experts {0, 1, 2}. */
    memset(w_gate, 0, sizeof(w_gate));
    dispatching = 1;
    CHECK(et_mod5_moe_forward_f32(n, t, d, f, e, k, x, w_gate, w_ge, w_ue, w_de,
                                  0.125, y, &aux, counts, &evals, &error) == 0);
    CHECK(evals == 9);
    CHECK(counts[0] == 3 && counts[1] == 3 && counts[2] == 3 &&
          counts[3] == 0 && counts[4] == 0 && counts[5] == 0);

    /* Partial tie with K=2: experts {1, 3, 5} tied at +2.0, {0, 2, 4} at -2.0.
     * Lowest-index tie-breaking MUST pick {1, 3} (never 5). */
    for (size_t ex = 0; ex < e; ++ex) {
      w_gate[ex * d + 0u] = (ex & 1u) ? 2.0f : -2.0f;
      w_gate[ex * d + 1u] = 0.0f;
    }
    CHECK(et_mod5_moe_forward_f32(n, t, d, f, e, 2u, x, w_gate, w_ge, w_ue,
                                  w_de, 0.125, y, &aux, counts, &evals,
                                  &error) == 0);
    dispatching = 0;
    CHECK(evals == 6);
    CHECK(counts[0] == 0 && counts[1] == 3 && counts[2] == 0 &&
          counts[3] == 3 && counts[4] == 0 && counts[5] == 0);
  }

  /* C. Verify dW_gate decomposes linearly into routing-weight + aux-loss terms,
   *    both non-zero for K=2, and route term == 0 for K=1. */
  {
    const size_t n = 1, t = 4, d = 4, f = 6, e = 4, k = 2;
    float x[16], w_gate[16], w_ge[96], w_ue[96], w_de[96], dy[16], dy_zero[16];
    float dx_r[16], dx_a[16], dx_b[16];
    float dwg_r[16], dwg_a[16], dwg_b[16];
    float dwge[96], dwue[96], dwde[96];

    fill_router_inputs(x, t, d, w_gate, e, 9);
    fill_deterministic_f32(w_ge, 96u, 59, 0.0625f);
    fill_deterministic_f32(w_ue, 96u, 61, 0.0625f);
    fill_deterministic_f32(w_de, 96u, 67, 0.0625f);
    fill_deterministic_f32(dy, 16u, 71, 0.125f);
    memset(dy_zero, 0, sizeof(dy_zero));

    dispatching = 1;
    /* Route-only (alpha_aux = 0) */
    CHECK(et_mod5_moe_backward_f32(n, t, d, f, e, k, x, w_gate, w_ge, w_ue,
                                   w_de, 0.0, dy, dx_r, dwg_r, dwge, dwue, dwde,
                                   &error) == 0);
    /* Aux-only (dY = 0, alpha_aux = 0.25) */
    CHECK(et_mod5_moe_backward_f32(n, t, d, f, e, k, x, w_gate, w_ge, w_ue,
                                   w_de, 0.25, dy_zero, dx_a, dwg_a, dwge, dwue,
                                   dwde, &error) == 0);
    /* Combined (dY != 0, alpha_aux = 0.25) */
    CHECK(et_mod5_moe_backward_f32(n, t, d, f, e, k, x, w_gate, w_ge, w_ue,
                                   w_de, 0.25, dy, dx_b, dwg_b, dwge, dwue,
                                   dwde, &error) == 0);

    double norm_r = 0.0, norm_a = 0.0;
    for (size_t i = 0; i < 16u; ++i) {
      norm_r += fabs((double)dwg_r[i]);
      norm_a += fabs((double)dwg_a[i]);
      CHECK(fabs((double)dwg_b[i] - ((double)dwg_r[i] + (double)dwg_a[i])) <=
            1e-6);
      CHECK(fabs((double)dx_b[i] - ((double)dx_r[i] + (double)dx_a[i])) <=
            1e-6);
    }
    CHECK(norm_r > 1e-4 && norm_a > 1e-4);

    /* For K=1 with alpha_aux = 0, renormalized weight w_0 == 1 -> dW_gate == 0 */
    CHECK(et_mod5_moe_backward_f32(n, t, d, f, e, 1u, x, w_gate, w_ge, w_ue,
                                   w_de, 0.0, dy, dx_r, dwg_r, dwge, dwue, dwde,
                                   &error) == 0);
    dispatching = 0;
    for (size_t i = 0; i < 16u; ++i)
      CHECK(dwg_r[i] == 0.0f);
  }
}

static void check_negative_suite(const et_kernel_runtime *runtime) {
  et_kernel_runtime *baseline = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);

  unsigned char raw_x[512] __attribute__((aligned(16)));
  unsigned char raw_wg[512] __attribute__((aligned(16)));
  unsigned char raw_wge[2048] __attribute__((aligned(16)));
  unsigned char raw_wue[2048] __attribute__((aligned(16)));
  unsigned char raw_wde[2048] __attribute__((aligned(16)));
  unsigned char raw_dy[512] __attribute__((aligned(16)));

  unsigned char raw_y[512] __attribute__((aligned(16)));
  unsigned char raw_aux[64] __attribute__((aligned(16)));
  unsigned char raw_cnt[128] __attribute__((aligned(16)));
  unsigned char raw_evals[64] __attribute__((aligned(16)));

  unsigned char raw_dx[512] __attribute__((aligned(16)));
  unsigned char raw_dwg[512] __attribute__((aligned(16)));
  unsigned char raw_dwge[2048] __attribute__((aligned(16)));
  unsigned char raw_dwue[2048] __attribute__((aligned(16)));
  unsigned char raw_dwde[2048] __attribute__((aligned(16)));

  unsigned char sent_y[512], sent_aux[64], sent_cnt[128], sent_evals[64];
  unsigned char sent_dx[512], sent_dwg[512], sent_dwge[2048];

  memset(raw_x, 0, sizeof(raw_x));
  memset(raw_wg, 0, sizeof(raw_wg));
  memset(raw_wge, 0, sizeof(raw_wge));
  memset(raw_wue, 0, sizeof(raw_wue));
  memset(raw_wde, 0, sizeof(raw_wde));
  memset(raw_dy, 0, sizeof(raw_dy));

  memset(raw_y, 0xa5, sizeof(raw_y));
  memset(raw_aux, 0x5a, sizeof(raw_aux));
  memset(raw_cnt, 0xc3, sizeof(raw_cnt));
  memset(raw_evals, 0x3c, sizeof(raw_evals));
  memset(raw_dx, 0x99, sizeof(raw_dx));
  memset(raw_dwg, 0x66, sizeof(raw_dwg));
  memset(raw_dwge, 0xaa, sizeof(raw_dwge));
  memset(raw_dwue, 0x55, sizeof(raw_dwue));
  memset(raw_dwde, 0x77, sizeof(raw_dwde));

  memcpy(sent_y, raw_y, sizeof(sent_y));
  memcpy(sent_aux, raw_aux, sizeof(sent_aux));
  memcpy(sent_cnt, raw_cnt, sizeof(sent_cnt));
  memcpy(sent_evals, raw_evals, sizeof(sent_evals));
  memcpy(sent_dx, raw_dx, sizeof(sent_dx));
  memcpy(sent_dwg, raw_dwg, sizeof(sent_dwg));
  memcpy(sent_dwge, raw_dwge, sizeof(sent_dwge));

  float *x = (float *)raw_x;
  float *wg = (float *)raw_wg;
  float *wge = (float *)raw_wge;
  float *wue = (float *)raw_wue;
  float *wde = (float *)raw_wde;
  float *dy = (float *)raw_dy;
  fill_deterministic_f32(x, 64u, 1, 0.0625f);
  fill_deterministic_f32(wg, 64u, 2, 0.0625f);
  fill_deterministic_f32(wge, 256u, 3, 0.0625f);
  fill_deterministic_f32(wue, 256u, 4, 0.0625f);
  fill_deterministic_f32(wde, 256u, 5, 0.0625f);
  fill_deterministic_f32(dy, 64u, 6, 0.0625f);

  float alpha_ok = 0.0625f;

  dispatching = 1;

  /* 1. Baseline runtime has no verified kernel.moe capability. */
  CHECK(dispatch_moe_forward_k1(baseline, 1, 2, 2, 2, 2, 1, x, wg, wge, wue,
                                wde, &alpha_ok, raw_y, raw_aux, raw_cnt,
                                raw_evals, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);

  /* 2. Invalid shapes and top_k > E. */
  const size_t bad_shapes[][6] = {
      {0, 1, 2, 2, 2, 1},  {3, 1, 2, 2, 2, 1},  {1, 0, 2, 2, 2, 1},
      {1, 33, 2, 2, 2, 1}, {1, 1, 1, 2, 2, 1},  {1, 1, 33, 2, 2, 1},
      {1, 1, 2, 1, 2, 1},  {1, 1, 2, 65, 2, 1}, {1, 1, 2, 2, 1, 1},
      {1, 1, 2, 2, 9, 1},  {1, 1, 2, 2, 2, 0},  {1, 1, 2, 2, 8, 5},
      {1, 1, 2, 2, 2, 3},  {1, 1, 2, 2, 2, 4},  {1, 1, 2, 2, 3, 4}};
  for (size_t i = 0; i < ARRAY_COUNT(bad_shapes); ++i) {
    const size_t *s = bad_shapes[i];
    CHECK(et_mod5_moe_forward_f32(s[0], s[1], s[2], s[3], s[4], s[5], x, wg,
                                  wge, wue, wde, 0.0625, raw_y, raw_aux,
                                  raw_cnt, raw_evals, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(et_mod5_moe_backward_f32(s[0], s[1], s[2], s[3], s[4], s[5], x, wg,
                                   wge, wue, wde, 0.0625, dy, raw_dx, raw_dwg,
                                   raw_dwge, raw_dwue, raw_dwde, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(dispatch_moe_forward_k1(runtime, s[0], s[1], s[2], s[3], s[4], s[5],
                                  x, wg, wge, wue, wde, &alpha_ok, raw_y,
                                  raw_aux, raw_cnt, raw_evals, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
  }

  /* 3. NULL pointers. */
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, NULL, wg, wge, wue, wde,
                                0.0625, raw_y, raw_aux, raw_cnt, raw_evals,
                                &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                NULL, raw_aux, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, NULL, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, NULL, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_backward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                 NULL, raw_dx, raw_dwg, raw_dwge, raw_dwue,
                                 raw_dwde, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_backward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                 dy, raw_dx, raw_dwg, raw_dwge, raw_dwue, NULL,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* 4. Misaligned pointers via void* (zero UBSan alignment UB). */
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, raw_x + 1, wg, wge, wue, wde,
                                0.0625, raw_y, raw_aux, raw_cnt, raw_evals,
                                &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y + 2, raw_aux, raw_cnt, raw_evals,
                                &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt + 4, raw_evals,
                                &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, raw_evals + 4,
                                &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_moe_forward_k1(runtime, 1, 2, 2, 2, 2, 1, raw_x + 1, wg, wge,
                                wue, wde, &alpha_ok, raw_y, raw_aux, raw_cnt,
                                raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_moe_backward_k1(runtime, 1, 2, 2, 2, 2, 1, x, wg, wge, wue,
                                 wde, &alpha_ok, dy, raw_dx + 2, raw_dwg,
                                 raw_dwge, raw_dwue, raw_dwde, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* 5. Aliasing (input-input, output-input, output-output, output-error). */
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, x, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                x, raw_aux, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_y, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, raw_cnt, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(dispatch_moe_backward_k1(runtime, 1, 2, 2, 2, 2, 1, x, wg, wge, wue,
                                 wde, &alpha_ok, dy, raw_dx, raw_dx, raw_dwge,
                                 raw_dwue, raw_dwde, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);

  /* 6. Negative and non-finite alpha_aux. */
  const double bad_alphas[] = {-1e-6, -0.01, -1.0, NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < ARRAY_COUNT(bad_alphas); ++i) {
    float af = (float)bad_alphas[i];
    CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                  bad_alphas[i], raw_y, raw_aux, raw_cnt,
                                  raw_evals, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(et_mod5_moe_backward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                   bad_alphas[i], dy, raw_dx, raw_dwg, raw_dwge,
                                   raw_dwue, raw_dwde, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(dispatch_moe_forward_k1(runtime, 1, 2, 2, 2, 2, 1, x, wg, wge, wue,
                                  wde, &af, raw_y, raw_aux, raw_cnt, raw_evals,
                                  &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }

  /* 7. Non-finite inputs/weights/dY and binary32 output overflow. */
  const float bad_vals[] = {NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < ARRAY_COUNT(bad_vals); ++i) {
    const float saved = x[2];
    x[2] = bad_vals[i];
    CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                  0.0625, raw_y, raw_aux, raw_cnt, raw_evals,
                                  &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    x[2] = saved;

    const float saved_dy = dy[1];
    dy[1] = bad_vals[i];
    CHECK(et_mod5_moe_backward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                   0.0625, dy, raw_dx, raw_dwg, raw_dwge,
                                   raw_dwue, raw_dwde, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    dy[1] = saved_dy;
  }

  /* Binary32 overflow during forward writeback (errno preserved). */
  {
    const float sx = x[0], swg = wge[0], swu = wue[0], swd = wde[0];
    x[0] = 1.0e20f;
    wge[0] = 1.0f;
    wue[0] = 1.0e20f;
    wde[0] = 1.0f;
    errno = EAGAIN;
    CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                  0.0625, raw_y, raw_aux, raw_cnt, raw_evals,
                                  &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(errno == EAGAIN);
    x[0] = sx;
    wge[0] = swg;
    wue[0] = swu;
    wde[0] = swd;
  }

  /* 8. Non-standard FP environments (FE_UPWARD, MXCSR FTZ, MXCSR DAZ). */
  fenv_t saved_env;
  CHECK(fegetenv(&saved_env) == 0);
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(dispatch_moe_backward_k1(runtime, 1, 2, 2, 2, 2, 1, x, wg, wge, wue,
                                 wde, &alpha_ok, dy, raw_dx, raw_dwg, raw_dwge,
                                 raw_dwue, raw_dwde, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(fesetenv(&saved_env) == 0);

  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_mod5_moe_forward_f32(1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde, 0.0625,
                                raw_y, raw_aux, raw_cnt, raw_evals, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(saved_mxcsr | 0x0040u); /* DAZ */
  CHECK(dispatch_moe_forward_k1(runtime, 1, 2, 2, 2, 2, 1, x, wg, wge, wue, wde,
                                &alpha_ok, raw_y, raw_aux, raw_cnt, raw_evals,
                                &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(saved_mxcsr);

  /* 9. Direct provider validate_call NULL check. */
  const et_kernel_provider_v1 *provider = et_mod5_moe_kernel_provider_v1();
  CHECK(provider->validate_call(NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* Verify EVERY output sentinel remained byte-for-byte untouched! */
  CHECK(memcmp(raw_y, sent_y, sizeof(raw_y)) == 0);
  CHECK(memcmp(raw_aux, sent_aux, sizeof(raw_aux)) == 0);
  CHECK(memcmp(raw_cnt, sent_cnt, sizeof(raw_cnt)) == 0);
  CHECK(memcmp(raw_evals, sent_evals, sizeof(raw_evals)) == 0);
  CHECK(memcmp(raw_dx, sent_dx, sizeof(raw_dx)) == 0);
  CHECK(memcmp(raw_dwg, sent_dwg, sizeof(raw_dwg)) == 0);
  CHECK(memcmp(raw_dwge, sent_dwge, sizeof(raw_dwge)) == 0);

  dispatching = 0;
  et_kernel_runtime_destroy(baseline);
}

static void check_bridge(void) {
  et_kernel_runtime *rt = NULL;
  CHECK(et_mod5_moe_aot_open(NULL) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_aot_open(&rt) == 0 && rt != NULL);

  unsigned char raw_x[64] __attribute__((aligned(16)));
  unsigned char raw_wg[64] __attribute__((aligned(16)));
  unsigned char raw_wge[256] __attribute__((aligned(16)));
  unsigned char raw_wue[256] __attribute__((aligned(16)));
  unsigned char raw_wde[256] __attribute__((aligned(16)));
  unsigned char raw_dy[64] __attribute__((aligned(16)));
  unsigned char raw_y[64] __attribute__((aligned(16)));
  unsigned char raw_aux[16] __attribute__((aligned(16)));
  unsigned char raw_cnt[64] __attribute__((aligned(16)));
  unsigned char raw_evals[16] __attribute__((aligned(16)));
  unsigned char raw_dx[64] __attribute__((aligned(16)));
  unsigned char raw_dwg[64] __attribute__((aligned(16)));
  unsigned char raw_dwge[256] __attribute__((aligned(16)));
  unsigned char raw_dwue[256] __attribute__((aligned(16)));
  unsigned char raw_dwde[256] __attribute__((aligned(16)));
  unsigned char sent_y[64];

  for (int64_t i = 0; i < 8; ++i) {
    et_mod5_moe_aot_store_f32(raw_x, i, 0.125 * (double)(i - 3));
    et_mod5_moe_aot_store_f32(raw_wg, i, 0.25 * (double)((i % 4) - 1));
    et_mod5_moe_aot_store_f32(raw_dy, i, 0.0625 * (double)(i + 1));
  }
  for (int64_t i = 0; i < 32; ++i) {
    et_mod5_moe_aot_store_f32(raw_wge, i, 0.0625 * (double)((i % 7) - 3));
    et_mod5_moe_aot_store_f32(raw_wue, i, 0.0625 * (double)((i % 5) - 2));
    et_mod5_moe_aot_store_f32(raw_wde, i, 0.0625 * (double)((i % 9) - 4));
  }
  et_mod5_moe_aot_store_i64(raw_cnt, 0, 99);
  CHECK(et_mod5_moe_aot_load_i64(raw_cnt, 0) == 99);

  dispatching = 1;
  CHECK(et_mod5_moe_aot_forward(rt, 1, 2, 4, 2, 4, 2, raw_x, raw_wg, raw_wge,
                                raw_wue, raw_wde, 0.0625, raw_y, raw_aux,
                                raw_cnt, raw_evals) == 0);
  CHECK(et_mod5_moe_aot_load_i64(raw_evals, 0) == 4);
  CHECK(et_mod5_moe_aot_load_f32(raw_aux, 0) > 0.0);
  CHECK(et_mod5_moe_aot_backward(rt, 1, 2, 4, 2, 4, 2, raw_x, raw_wg, raw_wge,
                                 raw_wue, raw_wde, 0.0625, raw_dy, raw_dx,
                                 raw_dwg, raw_dwge, raw_dwue, raw_dwde) == 0);

  /* Bridge negative checks: top_k > E, misaligned pointer, negative alpha, Inf. */
  memset(raw_y, 0x5a, sizeof(raw_y));
  memcpy(sent_y, raw_y, sizeof(sent_y));
  CHECK(et_mod5_moe_aot_forward(rt, 1, 2, 4, 2, 2, 3, raw_x, raw_wg, raw_wge,
                                raw_wue, raw_wde, 0.0625, raw_y, raw_aux,
                                raw_cnt, raw_evals) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mod5_moe_aot_forward(rt, 1, 2, 4, 2, 4, 2, raw_x + 1, raw_wg,
                                raw_wge, raw_wue, raw_wde, 0.0625, raw_y,
                                raw_aux, raw_cnt, raw_evals) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_moe_aot_forward(rt, 1, 2, 4, 2, 4, 2, raw_x, raw_wg, raw_wge,
                                raw_wue, raw_wde, -0.01, raw_y, raw_aux,
                                raw_cnt, raw_evals) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  et_mod5_moe_aot_store_f32(raw_x, 0, DBL_MAX);
  CHECK(et_mod5_moe_aot_forward(rt, 1, 2, 4, 2, 4, 2, raw_x, raw_wg, raw_wge,
                                raw_wue, raw_wde, 0.0625, raw_y, raw_aux,
                                raw_cnt, raw_evals) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(raw_y, sent_y, sizeof(raw_y)) == 0);
  dispatching = 0;

  et_mod5_moe_aot_close(rt);
}

static void print_hex_floats(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i)
    printf("%s%a", i ? " " : "", (double)buf[i]);
  putchar('\n');
}

static void emit_oracle(const et_kernel_runtime *runtime) {
  for (size_t c = 0; c < ARRAY_COUNT(k_cases); ++c) {
    const moe_case *mc = &k_cases[c];
    /* Keep oracle payload compact while covering all 6 diverse cases + tie case. */
    if (mc->n == 2 && mc->t == 32)
      continue;
    const size_t x_cnt = mc->n * mc->t * mc->d;
    const size_t wg_cnt = mc->e * mc->d;
    const size_t exp_cnt = mc->e * mc->f * mc->d;

    float *x = (float *)calloc(x_cnt, sizeof(float));
    float *wg = (float *)calloc(wg_cnt, sizeof(float));
    float *wge = (float *)calloc(exp_cnt, sizeof(float));
    float *wue = (float *)calloc(exp_cnt, sizeof(float));
    float *wde = (float *)calloc(exp_cnt, sizeof(float));
    float *dy = (float *)calloc(x_cnt, sizeof(float));
    float *y = (float *)calloc(x_cnt, sizeof(float));
    float *dx = (float *)calloc(x_cnt, sizeof(float));
    float *dwg = (float *)calloc(wg_cnt, sizeof(float));
    float *dwge = (float *)calloc(exp_cnt, sizeof(float));
    float *dwue = (float *)calloc(exp_cnt, sizeof(float));
    float *dwde = (float *)calloc(exp_cnt, sizeof(float));
    int64_t *counts = (int64_t *)calloc(mc->e, sizeof(int64_t));
    CHECK(x && wg && wge && wue && wde && dy && y && dx && dwg && dwge &&
          dwue && dwde && counts);

    fill_router_inputs(x, mc->n * mc->t, mc->d, wg, mc->e, mc->salt);
    fill_deterministic_f32(wge, exp_cnt, mc->salt + 31, 0.03125f);
    fill_deterministic_f32(wue, exp_cnt, mc->salt + 53, 0.03125f);
    fill_deterministic_f32(wde, exp_cnt, mc->salt + 79, 0.03125f);
    fill_deterministic_f32(dy, x_cnt, mc->salt + 101, 0.0625f);

    float alpha = mc->alpha_aux;
    float aux = 0.0f;
    int64_t evals = 0;
    et_kernel_error error;
    CHECK(dispatch_moe_forward_k1(runtime, mc->n, mc->t, mc->d, mc->f, mc->e,
                                  mc->k, x, wg, wge, wue, wde, &alpha, y, &aux,
                                  counts, &evals, &error) == 0);
    CHECK(dispatch_moe_backward_k1(runtime, mc->n, mc->t, mc->d, mc->f, mc->e,
                                   mc->k, x, wg, wge, wue, wde, &alpha, dy, dx,
                                   dwg, dwge, dwue, dwde, &error) == 0);

    printf("CASE %zu %zu %zu %zu %zu %zu %a %a %lld\n", mc->n, mc->t, mc->d,
           mc->f, mc->e, mc->k, (double)alpha, (double)aux, (long long)evals);
    for (size_t e = 0; e < mc->e; ++e)
      printf("%s%lld", e ? " " : "", (long long)counts[e]);
    putchar('\n');
    print_hex_floats(x, x_cnt);
    print_hex_floats(wg, wg_cnt);
    print_hex_floats(wge, exp_cnt);
    print_hex_floats(wue, exp_cnt);
    print_hex_floats(wde, exp_cnt);
    print_hex_floats(dy, x_cnt);
    print_hex_floats(y, x_cnt);
    print_hex_floats(dx, x_cnt);
    print_hex_floats(dwg, wg_cnt);
    print_hex_floats(dwge, exp_cnt);
    print_hex_floats(dwue, exp_cnt);
    print_hex_floats(dwde, exp_cnt);

    free(x);
    free(wg);
    free(wge);
    free(wue);
    free(wde);
    free(dy);
    free(y);
    free(dx);
    free(dwg);
    free(dwge);
    free(dwue);
    free(dwde);
    free(counts);
  }
}

int main(int argc, char **argv) {
  et_kernel_runtime *runtime = discover_moe();
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    emit_oracle(runtime);
    et_kernel_runtime_destroy(runtime);
    return 0;
  }

  /* Verify K1 capability metadata. */
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, "kernel.moe");
  CHECK(cap != NULL);
  CHECK(cap->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(cap->deterministic == 1u);
  CHECK(cap->operation_count == 2u);
  CHECK(cap->shape_range_count == 4u);

  double max_fd_err = 0.0;
  size_t fd_checks = 0u;
  for (size_t i = 0; i < ARRAY_COUNT(k_cases); ++i)
    run_case(runtime, &k_cases[i], &max_fd_err, &fd_checks);

  check_sparse_skip_tie_breaking_and_router_decomp(runtime);
  check_negative_suite(runtime);
  check_bridge();

  et_kernel_runtime_destroy(runtime);
  printf("MOD5 MoE native suite: PASS (cases=%zu fd_checks=%zu "
         "max_fd_err=%.3e)\n",
         ARRAY_COUNT(k_cases), fd_checks, max_fd_err);
  return 0;
}
