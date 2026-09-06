#include "d2_native.h"

#include <cstdint>
#include <type_traits>

static_assert(std::is_same_v<decltype(et_d2_batch_last_status_v1()), std::int64_t>);
static_assert(std::is_same_v<decltype(et_d2_dataset_open_v1(nullptr)), std::int64_t>);
static_assert(std::is_same_v<decltype(et_d2_dataset_close_v1(nullptr)), std::int64_t>);
static_assert(std::is_same_v<decltype(et_d2_batch_release_preflight_v1(nullptr, 0)),
                             std::int64_t>);
static_assert(std::is_same_v<decltype(et_d2_batch_write_pair_i64le_source_span_v1(
                                 nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0,
                                 0)),
                             std::int64_t>);

int main() { return ET_D2_NATIVE_STATUS_OK; }
