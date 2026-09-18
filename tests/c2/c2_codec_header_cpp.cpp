#include "c2_checkpoint_codec.h"

#include <type_traits>

static_assert(ET_C2_CHECKPOINT_ENCODE_REQUEST_V1_SIZE ==
              sizeof(et_c2_checkpoint_encode_request_v1), "request ABI");
static_assert(std::is_same_v<decltype(&et_c2_checkpoint_encode_measure_v1),
                             int32_t (*)(
                                 const et_c2_checkpoint_encode_request_v1 *,
                                 size_t *,
                                 et_c2_checkpoint_format_error_v1 *)>);
static_assert(std::is_same_v<decltype(&et_c2_checkpoint_encode_v1),
                             int32_t (*)(
                                 const et_c2_checkpoint_encode_request_v1 *,
                                 uint8_t *, size_t,
                                 et_c2_checkpoint_format_error_v1 *)>);

int main() {
  et_c2_checkpoint_encode_request_v1 request{};
  request.struct_size = sizeof(request);
  return request.struct_size == ET_C2_CHECKPOINT_ENCODE_REQUEST_V1_SIZE ? 0 : 1;
}
