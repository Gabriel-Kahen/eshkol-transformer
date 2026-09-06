#include "eshkol_transformer/indexed_cross_entropy.h"

int main() {
  return et_l2_indexed_cross_entropy_abi_major_v1() == 1 &&
                 et_l2_indexed_cross_entropy_provider_v1() != nullptr
             ? 0
             : 1;
}
