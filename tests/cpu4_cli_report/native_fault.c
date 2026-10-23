/* Standalone host C logic fixture with explicit F32/allocation doubles.
 * Compile this TU alone; do not link it alongside the actual production shim.
 * No production testing flag, observer symbol, or fault control is added. */
#define calloc cli_fault_calloc
#define et_f32_tensor_create_v1 fault_create
#define et_f32_tensor_private_destroy_ready_v1 fault_ready
#define et_f32_tensor_rank_v1 fault_rank
#define et_f32_tensor_element_count_v1 fault_elements
#define et_f32_tensor_byte_length_v1 fault_bytes
#define et_f32_tensor_scoped_begin_internal fault_begin
#define et_f32_tensor_scoped_end_internal fault_end
#define et_f32_tensor_copy_bits_to_v1 fault_copy
#define et_f32_tensor_destroy_v1 fault_destroy
#include "../../native/cpu4_cli_package_bridge.c"
#undef calloc
#include <assert.h>
#include <stdio.h>
static int allocation_fault, failing_call, diagnostic, call_index, scope_count;
static unsigned owner_index;
static uint64_t tokens[32];
void *cli_fault_calloc(size_t n, size_t size) {
  if (allocation_fault) return NULL;
  void *p = malloc(n * size); if (p) memset(p, 0, n * size); return p;
}
static int status(et_f32_tensor_error *error) {
  ++call_index;
  /* The actual shim must clear EVERY sink before each call. */
  assert(error->category == 0 && error->code == 0);
  if (call_index != failing_call) return 0;
  if (diagnostic == 1) { error->category = 4; error->code = 8; }
  if (diagnostic == 2) { error->category = 1; error->code = 8; }
  if (diagnostic == 3) { error->category = 4; error->code = 12; }
  return 4;
}
int32_t fault_create(size_t rank, const uint64_t *shape, et_f32_tensor **owner,
                     et_f32_tensor_error *error) {
  assert(rank == 0 && shape == NULL && *owner == NULL);
  int s = status(error); if (!s) *owner = (et_f32_tensor *)&tokens[owner_index++]; return s;
}
int32_t fault_ready(const et_f32_tensor *owner, et_f32_tensor_error *error) {
  assert(owner); return status(error);
}
int32_t fault_rank(const et_f32_tensor *owner, size_t *out, et_f32_tensor_error *error) {
  (void)owner; int s = status(error); if (!s) *out = 0; return s;
}
int32_t fault_elements(const et_f32_tensor *owner, size_t *out, et_f32_tensor_error *error) {
  (void)owner; int s = status(error); if (!s) *out = 1; return s;
}
int32_t fault_bytes(const et_f32_tensor *owner, size_t *out, et_f32_tensor_error *error) {
  (void)owner; int s = status(error); if (!s) *out = 4; return s;
}
int32_t fault_begin(et_f32_tensor *owner, et_f32_scoped_guard_internal *guard,
                     et_f32_tensor_error *error) {
  int s = status(error); if (s) return s;
  assert(guard->owner == NULL && guard->self == NULL);
  guard->owner = owner; guard->self = guard;
  guard->view = (et_kernel_tensor_view_v1){sizeof(guard->view), tokens, 4,
    "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 0, NULL};
  ++scope_count; return 0;
}
int32_t fault_end(et_f32_scoped_guard_internal *guard) {
  assert(guard->owner && guard->self == guard && scope_count == 1);
  --scope_count; guard->owner = NULL; guard->self = NULL; return 0;
}
int32_t fault_copy(const et_f32_tensor *owner, uint32_t *out, size_t count,
                    et_f32_tensor_error *error) {
  assert(owner && out && count == 1 && scope_count == 1);
  int s = status(error); if (!s) *out = 0x7fc12345; return s;
}
int32_t fault_destroy(et_f32_tensor **owner, et_f32_tensor_error *error) {
  assert(owner && *owner && scope_count == 0);
  int s = status(error); if (!s) *owner = NULL; return s;
}
static void reset_fault(int call, int diag) { call_index = 0; failing_call = call; diagnostic = diag; }
static void expect_error(int domain, int category, int code) {
  assert(et_cpu4_private_cli_scalar_last_error_ref_v1(0) == domain);
  assert(et_cpu4_private_cli_scalar_last_error_ref_v1(1) == category);
  assert(et_cpu4_private_cli_scalar_last_error_ref_v1(2) == code);
}
int main(void) {
  allocation_fault = 1;
  assert(et_cpu4_private_cli_scalar_create_v1() == NULL); expect_error(0, 5, 5);
  allocation_fault = 0;
  for (int diag = 0; diag < 4; ++diag) {
    reset_fault(1, diag);
    assert(et_cpu4_private_cli_scalar_create_v1() == NULL);
    if (diag == 1) expect_error(1, 4, 8); else expect_error(0, 6, 6);
  }
  reset_fault(0, 0);
  et_f32_tensor *scalar = et_cpu4_private_cli_scalar_create_v1(); assert(scalar);
  /* ready, rank, elements, bytes, scope begin, physical copy cuts. */
  for (int cut = 1; cut <= 6; ++cut) for (int diag = 0; diag < 4; ++diag) {
    reset_fault(cut, diag);
    assert(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == -1);
    assert(scope_count == 0);
    if (diag == 1) expect_error(1, 4, 8); else expect_error(0, 6, 6);
  }
  reset_fault(0, 0);
  assert(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == 0x7fc12345); expect_error(0, 0, 0);
  /* A rejected destroy retains exact owner; same-owner release retries. */
  reset_fault(6, 1);
  assert(et_cpu4_private_cli_scalar_release_v1(scalar) == -1); expect_error(1, 4, 8);
  reset_fault(0, 0);
  assert(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == 0x7fc12345);
  assert(et_cpu4_private_cli_scalar_release_v1(scalar) == 0);
  assert(et_cpu4_private_cli_scalar_release_v1(scalar) == 0);
  puts("CPU4-CLI-SCALAR-C-DOUBLES-PASS");
  return 0;
}
