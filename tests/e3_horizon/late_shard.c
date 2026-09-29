/* Test-only filesystem fault: remove the final D1 shard after D2 opens. */
#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char shard[PATH_MAX];
static char held[PATH_MAX];
static int hidden;

int64_t et_e3_horizon_hide_late_shard_v1(void) {
  const char *configured = getenv("E3_HORIZON_LATE_SHARD");
  if (hidden || !configured || configured[0] != '/') return 1;
  int original_bytes = snprintf(shard, sizeof shard, "%s", configured);
  int held_bytes = snprintf(held, sizeof held, "%s.held", configured);
  if (original_bytes < 0 || original_bytes >= (int)sizeof shard ||
      held_bytes < 0 || held_bytes >= (int)sizeof held ||
      rename(shard, held) != 0) return 1;
  hidden = 1;
  return 0;
}

int64_t et_e3_horizon_restore_late_shard_v1(void) {
  if (!hidden || rename(held, shard) != 0) return 1;
  hidden = 0;
  return 0;
}
