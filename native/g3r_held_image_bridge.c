/* Test-aggregate-only bridge from an Eshkol bytevector to the held G3 owner.
 * The Eshkol coordinator, not this bridge, authenticates its outer ledger. */
#if !defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE) || \
    !defined(ET_G3R_COMMITTED_HISTORY_PRIVATE)
#error "held image bridge requires the private current-history owner"
#endif
#include "eshkol_transformer/g3t_transport.h"

#include <stdint.h>
#include <string.h>

int64_t et_g3r_private_held_image_copy_v1(void *hold, void *header) {
  int64_t length = 0;
  if (!header) return -1;
  memcpy(&length, header, sizeof(length));
  if (length != (int64_t)sizeof(et_g3r_g3t_held_words_v1)) return -1;
  et_g3r_g3t_held_words_v1 *image =
      (et_g3r_g3t_held_words_v1 *)((unsigned char *)header + sizeof(length));
  return et_g3r_private_g3t_save_copy_v1(hold, image, length);
}
