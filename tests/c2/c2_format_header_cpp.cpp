#include "c2_checkpoint_format.h"

static_assert(ET_C2_CHECKPOINT_FORMAT_ABI_MAJOR == 1u, "ABI major changed");

int main() {
  et_c2_checkpoint_limits_v1 limits{};
  et_c2_checkpoint_view_v1 view{};
  et_c2_checkpoint_format_error_v1 error{};
  limits.struct_size = sizeof(limits);
  view.struct_size = sizeof(view);
  error.struct_size = sizeof(error);
  return 0;
}
