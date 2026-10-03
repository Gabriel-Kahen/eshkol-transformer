#define ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE 1
#define ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE 1
#define ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_PRIVATE 1
#define ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN p2g1_predecessor_main
#include "test_p2_g1_publication.c"

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  publication_success(owner, 0);
  publication_success(owner, 1);
  printf("G3-C4 P2/G1 with terminal feature PASS: checks=%zu\n", checks);
  return 0;
}
