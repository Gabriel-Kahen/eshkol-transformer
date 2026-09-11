#include "c2_checkpoint_reader.h"

#include <type_traits>

static_assert(
    std::is_same_v<decltype(&et_c2_checkpoint_reader_open_v1),
                   int64_t (*)(const char *, et_c2_checkpoint_reader **,
                               uint64_t *)>);
static_assert(std::is_same_v<decltype(&et_c2_checkpoint_reader_read_exact_v1),
                             int64_t (*)(et_c2_checkpoint_reader *, uint64_t,
                                         void *, uint64_t)>);
static_assert(
    std::is_same_v<decltype(&et_c2_checkpoint_reader_validate_final_v1),
                   int64_t (*)(et_c2_checkpoint_reader *)>);
static_assert(std::is_same_v<decltype(&et_c2_checkpoint_reader_close_v1),
                             int64_t (*)(et_c2_checkpoint_reader **)>);

int main() { return 0; }
