#include "checkpoint_io.h"

#include <cstdint>
#include <type_traits>

static_assert(ET_CHECKPOINT_IO_ABI_MAJOR == 1u);
static_assert(ET_CHECKPOINT_IO_ABI_MINOR == 0u);
static_assert(
    std::is_same_v<decltype(et_checkpoint_io_abi_major_v1()), std::int64_t>);
static_assert(std::is_same_v<
              decltype(et_checkpoint_io_read_exact_v1(nullptr, nullptr, 0, 1)),
              std::int64_t>);

int main() {
  return et_checkpoint_io_status_stage_v1(0) == ET_CHECKPOINT_IO_STAGE_NONE ? 0
                                                                            : 1;
}
