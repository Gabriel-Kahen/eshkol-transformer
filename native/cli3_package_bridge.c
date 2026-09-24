#include <errno.h>
#include <stdint.h>
#include <unistd.h>

#include "e1b_error_consumer_bridge.h"

/* Supply the predecessor's private native identities in this one aggregate.
 * Its public wrappers are localized by the CLI3 export allowlist. */
#include "c2_wave2_package_bridge.c"

extern eshkol_tagged_value_t cli3_private_dispatch_cabi_v1(
    eshkol_tagged_value_t arguments);

static int64_t cli3_fallback_status = 70;

void et_cli3_private_fallback_status_set_v1(int64_t status) {
  cli3_fallback_status = status;
}

int64_t et_cli3_private_stdout_write_v1(const char *text, int64_t length) {
  if (text == NULL || length < 0) {
    return EINVAL;
  }
  while (length != 0) {
    ssize_t amount = write(STDOUT_FILENO, text, (size_t)length);
    if (amount > 0) {
      text += (size_t)amount;
      length -= amount;
    } else if (amount < 0 && errno == EINTR) {
      continue;
    } else {
      return errno != 0 ? errno : EIO;
    }
  }
  return 0;
}

void et_e1b_public_cli3_dispatch_v1(void *arguments, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      cli3_private_dispatch_cabi_v1(*et_e1b_box_value_v1(arguments));
}

static int cli3_write_all(const char *text, size_t length) {
  while (length != 0u) {
    ssize_t amount = write(STDERR_FILENO, text, length);
    if (amount > 0) {
      text += (size_t)amount;
      length -= (size_t)amount;
    } else if (amount < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return 0;
}

int64_t et_e1b_public_cli3_dispatch_fallback_v1(void) {
  static const char usage[] =
      "eshkol-transformer: usage: invalid command line\n";
  static const char io[] =
      "eshkol-transformer: io: publication state unknown; inspect the output path\n";
  static const char internal[] =
      "eshkol-transformer: internal: diagnostic unavailable\n";
  const char *text = internal;
  size_t length = sizeof(internal) - 1u;
  int64_t status = 70;

  if (cli3_fallback_status == 2) {
    text = usage;
    length = sizeof(usage) - 1u;
    status = 2;
  } else if (cli3_fallback_status == 13) {
    text = io;
    length = sizeof(io) - 1u;
    status = 13;
  }
  (void)cli3_write_all(text, length);
  return status;
}
