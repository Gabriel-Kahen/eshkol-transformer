/* The single proposed CPU4 CLI package/IO shim. No numerical owner/provider.
 * Calls are serialized by the closed Scheme leaf; private linkage only. */
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#include "../src/eshkol_transformer/cpu4_cli_scalar_private.h"
#include "../src/eshkol_transformer/m3t_f32_scoped.h"
#include <stdlib.h>
#include <string.h>

typedef struct cli_scalar_custody {
  struct cli_scalar_custody *next;
  et_f32_tensor *original;
  uint32_t state, claimed;
  uint64_t reserved0;
} cli_scalar_custody;
_Static_assert(sizeof(void *) == 8 && sizeof(cli_scalar_custody) == 32,
               "CLI scalar custody requires LP64/32 bytes");
static cli_scalar_custody *cli_scalars;
static _Thread_local int64_t cli_error[3];
static void cli_reset(void) { memset(cli_error, 0, sizeof(cli_error)); }
static int cli_fail(int condition) {
  cli_error[0] = 0; cli_error[1] = condition; cli_error[2] = condition;
  return -1;
}
static int cli_native(int32_t status, const et_f32_tensor_error *error) {
  if (status == 0) return 0;
  if (status >= 1 && status <= 6 && error->category == (uint32_t)status &&
      error->code >= 1 && error->code <= 11) {
    cli_error[0] = 1; cli_error[1] = status; cli_error[2] = error->code;
    return -1;
  }
  return cli_fail(6);
}
/* Lookup always precedes input dereference, including DEAD and foreign values. */
static cli_scalar_custody *cli_find(const et_f32_tensor *candidate) {
  for (cli_scalar_custody *row = cli_scalars; row; row = row->next)
    if (row->original == candidate) return row;
  cli_fail(1); return NULL;
}
/* Every native error sink is zeroed immediately before its call. */
#define CLI_CALL(expression) do { memset(&error, 0, sizeof(error)); \
  if (cli_native((expression), &error)) return -1; } while (0)
static int cli_scope(cli_scalar_custody *row, et_f32_scoped_guard_internal *scope) {
  et_f32_tensor_error error = {0};
  size_t rank = SIZE_MAX, elements = 0, bytes = 0;
  if (row->state != 1 || row->claimed > 1 || row->reserved0 != 0)
    return cli_fail(2);
  CLI_CALL(et_f32_tensor_private_destroy_ready_v1(row->original, &error));
  CLI_CALL(et_f32_tensor_rank_v1(row->original, &rank, &error));
  CLI_CALL(et_f32_tensor_element_count_v1(row->original, &elements, &error));
  CLI_CALL(et_f32_tensor_byte_length_v1(row->original, &bytes, &error));
  if (rank != 0 || elements != 1 || bytes != 4) return cli_fail(3);
  CLI_CALL(et_f32_tensor_scoped_begin_internal(row->original, scope, &error));
  const et_kernel_tensor_view_v1 *view = &scope->view;
  if (view->struct_size != sizeof(*view) || !view->data ||
      view->byte_length != 4 || !view->dtype || strcmp(view->dtype, "f32") ||
      !view->device || strcmp(view->device, "cpu") ||
      view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0 || view->rank != 0 || view->shape != NULL) {
    if (et_f32_tensor_scoped_end_internal(scope) != 0) abort();
    return cli_fail(3);
  }
  return 0;
}
static void cli_end(et_f32_scoped_guard_internal *scope) {
  if (et_f32_tensor_scoped_end_internal(scope) != 0) abort();
}
et_f32_tensor *et_cpu4_private_cli_scalar_create_v1(void) {
  cli_reset();
  cli_scalar_custody *row = calloc(1, sizeof(*row));
  et_f32_tensor_error error = {0};
  et_f32_tensor *owner = NULL;
  if (!row) { cli_fail(5); return NULL; }
  int32_t status = et_f32_tensor_create_v1(0, NULL, &owner, &error);
  if (cli_native(status, &error)) { free(row); return NULL; }
  /* Native create owns failure atomicity; success enrolls using fixed stores. */
  row->original = owner; row->state = 1; row->next = cli_scalars;
  cli_scalars = row;
  return owner;
}
int64_t et_cpu4_private_cli_scalar_read_bits_v1(const et_f32_tensor *scalar) {
  cli_reset();
  cli_scalar_custody *row = cli_find(scalar);
  et_f32_scoped_guard_internal scope = {0};
  et_f32_tensor_error error = {0};
  uint32_t bits = 0;
  if (!row || cli_scope(row, &scope)) return -1;
  int32_t status = et_f32_tensor_copy_bits_to_v1(row->original, &bits, 1, &error);
  int failed = cli_native(status, &error);
  cli_end(&scope);
  return failed ? -1 : (int64_t)bits;
}
int64_t et_cpu4_private_cli_scalar_claim_v1(et_f32_tensor *scalar, int64_t action) {
  cli_reset();
  if (action != 1 && action != 2) return cli_fail(4);
  cli_scalar_custody *row = cli_find(scalar);
  et_f32_scoped_guard_internal scope = {0};
  if (!row || cli_scope(row, &scope)) return -1;
  cli_end(&scope);
  if (row->claimed != (action == 2)) return cli_fail(2);
  row->claimed = action == 1;
  return 0;
}
int64_t et_cpu4_private_cli_scalar_release_v1(et_f32_tensor *scalar) {
  cli_reset();
  cli_scalar_custody *row = cli_find(scalar);
  et_f32_scoped_guard_internal scope = {0};
  et_f32_tensor_error error = {0};
  if (!row) return -1;
  if (row->state == 2 && row->claimed == 0 && row->reserved0 == 0) return 0;
  if (row->claimed) return cli_fail(2);
  if (cli_scope(row, &scope)) return -1;
  cli_end(&scope);
  et_f32_tensor *local = row->original;
  CLI_CALL(et_f32_tensor_destroy_v1(&local, &error));
  row->state = 2; row->claimed = 0;
  return 0;
}
int64_t et_cpu4_private_cli_scalar_last_error_ref_v1(int64_t selector) {
  return selector >= 0 && selector < 3 ? cli_error[selector] : -1;
}
