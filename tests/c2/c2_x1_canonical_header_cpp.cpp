#include "c2_x1_canonical.h"
#include <type_traits>
static_assert(std::is_standard_layout<et_c2_x1_projection_v1>::value, "ABI");
int main() { return ET_C2_X1_MAX_CANONICAL_BYTES == 16384u ? 0 : 1; }
