#include "d2_native.h"

#include <cstdint>
#include <type_traits>

static_assert(std::is_same_v<decltype(et_d2_batch_last_status_v1()), std::int64_t>);

int main() { return ET_D2_NATIVE_STATUS_OK; }

