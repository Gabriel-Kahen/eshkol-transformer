#include "mod5_v256_score_bridge.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float logits[4][8 * 256];
static uint32_t tokens[18];
static double metrics[5];
static int64_t counts[2];
static float probes[8];
static const uint8_t train_bytes[18] = {
    1, 4, 1, 5, 9, 2, 6, 5, 3, 2, 7, 1, 8, 2, 8, 1, 8, 2};
static const uint8_t eval_bytes[18] = {
    200, 3, 10, 11, 12, 13, 14, 15, 201,
    201, 15, 14, 13, 12, 11, 10, 3, 200};

static int32_t score(void) {
  return et_eval5_v256_score(logits[0], logits[1], logits[2], logits[3],
                             tokens, metrics, counts, probes);
}

int main(void) {
  uint32_t train_ids[18], eval_ids[18];
  uint64_t audit[8];
  for (size_t i = 0; i < 18u; ++i) {
    train_ids[i] = train_bytes[i];
    eval_ids[i] = eval_bytes[i];
  }
  const uint64_t clean[8] = {2, 0, 0, 0, 14, 0, 0, 0};
  const uint64_t contaminated[8] = {2, 2, 2, 2, 14, 14, 0, 1};
  if (et_eval5_v256_audit(train_bytes, train_ids, eval_bytes, eval_ids,
                          audit) != ET_EVAL5_OK ||
      memcmp(audit, clean, sizeof(clean)) != 0)
    return 6;
  if (et_eval5_v256_audit(train_bytes, train_ids, train_bytes, train_ids,
                          audit) != ET_EVAL5_ERR_CONTAMINATED ||
      memcmp(audit, contaminated, sizeof(contaminated)) != 0)
    return 7;
  audit[0] = 99;
  eval_ids[0]++;
  if (et_eval5_v256_audit(train_bytes, train_ids, eval_bytes, eval_ids,
                          audit) != ET_EVAL5_ERR_IDENTITY || audit[0] != 99)
    return 8;
  eval_ids[0]--;
  if (et_eval5_v256_audit(NULL, train_ids, eval_bytes, eval_ids,
                          audit) != ET_EVAL5_ERR_ARGUMENT || audit[0] != 99)
    return 9;
  for (size_t i = 0; i < 18u; ++i) tokens[i] = (uint32_t)i;
  if (score() != ET_EVAL5_OK ||
      fabs(metrics[0] - 16.0 * log(256.0)) > 1e-12 ||
      fabs(metrics[1] - log(256.0)) > 1e-12 ||
      fabs(metrics[2] - 256.0) > 1e-10 ||
      fabs(metrics[3] - 8.0) > 1e-12 ||
      counts[0] != 2 || counts[1] != 4 || metrics[4] != 0.5)
    return 1;
  if (et_eval5_v256_metric(metrics, 0) != metrics[0] ||
      et_eval5_v256_metric(metrics, 4) != metrics[4] ||
      !isnan(et_eval5_v256_metric(metrics, -1)) ||
      !isnan(et_eval5_v256_metric(metrics, 5)) ||
      !isnan(et_eval5_v256_metric(NULL, 0)))
    return 5;

  metrics[0] = -123.0;
  counts[0] = -123;
  probes[0] = -123.0f;
  tokens[1] = 256u;
  if (score() != ET_EVAL5_ERR_TARGET || metrics[0] != -123.0 ||
      counts[0] != -123 || probes[0] != -123.0f)
    return 2;
  tokens[1] = 1u;
  logits[2][0] = INFINITY;
  if (score() != ET_EVAL5_ERR_NONFINITE || metrics[0] != -123.0 ||
      counts[0] != -123 || probes[0] != -123.0f)
    return 3;
  logits[2][0] = 0.0f;
  if (et_eval5_v256_score(NULL, logits[1], logits[2], logits[3], tokens,
                          metrics, counts, probes) != ET_EVAL5_ERR_ARGUMENT)
    return 4;
  puts("EVAL5 V256 C audit/scorer bridge: PASS");
  return 0;
}
