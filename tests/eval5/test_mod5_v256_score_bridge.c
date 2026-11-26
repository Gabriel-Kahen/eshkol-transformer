#include "mod5_v256_score_bridge.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static float logits[4][8 * 256];
static uint32_t tokens[18];
static double metrics[5];
static int64_t counts[2];
static float probes[8];

static int32_t score(void) {
  return et_eval5_v256_score(logits[0], logits[1], logits[2], logits[3],
                             tokens, metrics, counts, probes);
}

int main(void) {
  for (size_t i = 0; i < 18u; ++i) tokens[i] = (uint32_t)i;
  if (score() != ET_EVAL5_OK ||
      fabs(metrics[0] - 16.0 * log(256.0)) > 1e-12 ||
      fabs(metrics[1] - log(256.0)) > 1e-12 ||
      fabs(metrics[2] - 256.0) > 1e-10 ||
      fabs(metrics[3] - 8.0) > 1e-12 ||
      counts[0] != 2 || counts[1] != 4 || metrics[4] != 0.5)
    return 1;

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
  puts("EVAL5 V256 C scorer bridge: PASS");
  return 0;
}
