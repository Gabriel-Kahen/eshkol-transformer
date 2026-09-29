/* Test-only E3/G3-T composition: one M3 registry and owner for both facets. */
#ifndef ET_CLI3_E3_G3T_AGGREGATE_BUILD
#error "CLI3 E3/G3-T aggregate requires its exact package build mode"
#endif
#include "../../src/eshkol_transformer/e3_frame.c"
#define ET_CLI3_E3_M3_OWNER_INCLUDED 1
#define ET_G3T_REUSE_E3_M3_OWNER_PRIVATE 1
#include "../../src/eshkol_transformer/g3t_transport.c"
