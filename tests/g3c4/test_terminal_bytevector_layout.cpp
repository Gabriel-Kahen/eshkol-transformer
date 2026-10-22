// Compile with the pinned Eshkol runtime_bytevector.cpp and inc/ directory.
#include <eshkol/eshkol.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" void *arena_allocate_with_header(void *, uint64_t bytes,
                                               uint8_t, uint8_t) {
  return std::calloc(1, bytes);
}
extern "C" void eshkol_runtime_fatal(eshkol_exception_type_t, const char *, ...) {
  std::abort();
}
extern "C" void *eshkol_make_bytevector(void *, int64_t, int64_t);
extern "C" void eshkol_bytevector_u8_set(void *, int64_t, int64_t);
extern "C" int64_t eshkol_bytevector_length(void *);

static int64_t declared(void *bytes) {
  int64_t value;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

int main() {
  for (int64_t payload : {int64_t{8}, int64_t{16}, int64_t{64}}) {
    void *correct = eshkol_make_bytevector(reinterpret_cast<void *>(1), payload, 0);
    assert(declared(correct) == payload);
    assert(eshkol_bytevector_length(correct) == payload);
    eshkol_bytevector_u8_set(correct, 0, 42);
    assert(declared(correct) == payload);
    assert(static_cast<uint8_t *>(correct)[8] == 42);
    std::free(correct);

    void *old = eshkol_make_bytevector(reinterpret_cast<void *>(1), payload + 8, 0);
    eshkol_bytevector_u8_set(old, 0, payload);
    assert(declared(old) == payload + 8);  // Native shape check rejects it.
    assert(static_cast<uint8_t *>(old)[8] == payload);
    std::free(old);
  }
}
