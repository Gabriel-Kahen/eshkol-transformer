#include <stdint.h>

/* Physical I2 parameter copies are compared to little-endian V1 words. */
int64_t et_g3r_private_fresh_host_le_v1(void) {
  const uint32_t one = 1u;
  return *(const uint8_t *)&one == 1u ? 1 : 0;
}
