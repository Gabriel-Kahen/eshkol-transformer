#include "e1b_error_consumer_bridge.h"

/* D2 supplies the single T2/I2/P1 identity universe. Its existing public
 * persistence symbol is retargeted to the C2 policy constructor. */
#define ET_E1B_PERSISTENCE_POLICY_PRIVATE_TARGET \
  et_e1b_private_c2_persistence_policy_cabi_v1
#define ET_C2_I2_MODEL_COPY 1
#include "d2_wave2_package_bridge.c"
#undef ET_C2_I2_MODEL_COPY
#undef ET_E1B_PERSISTENCE_POLICY_PRIVATE_TARGET

/* O2 extends the same I2 universe and exposes both its public suffix and the
 * C2-only reconstruction/copy seams without compiling another I2 bridge. */
#define ET_O2_NATIVE_HELPERS_ONLY 1
#define ET_C2_O2_RECONSTRUCT_BRIDGE 1
#include "o2_wave2_package_bridge.c"
#undef ET_C2_O2_RECONSTRUCT_BRIDGE
#undef ET_O2_NATIVE_HELPERS_ONLY
#include "o2_wave2_public_bridge_extension.c"

#include "k2_wave2_public_wrappers.inc"
#include "c2_wave2_public_wrappers.inc"
