/* Development-only differential access to source-private helpers; no test ABI. */
#define ET_F32_TENSOR_TESTING 1
#include "../../native/f32_tensor.c"

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "alias envelope FAIL line %d: %s\n", __LINE__, #x); \
  exit(1); } } while (0)

static void differential(const void *p, size_t n) {
  CHECK(storage_aliases_live(p, n) == storage_aliases_live_reference(p, n));
}

/* Explicit contract inventory, independent of the shortcut's allocation recorder. */
enum allocation_class {
  TENSOR_CONTROL, TENSOR_SHAPE, TENSOR_STRIDES, TENSOR_PAYLOAD,
  BORROW_CONTROL, COPY_CONTROL, COPY_ASSIGNMENTS, PARAMETER_CONTROL,
  GRADIENT_CONTROL, GRADIENT_ENTRIES, GRADIENT_PREPARED,
  RESET_CONTROL, RESET_PARAMETERS, LIVE_CLASS_COUNT
};
static unsigned covered;

static void protected_span(const void *p, size_t n) {
  uintptr_t first = (uintptr_t)p;
  CHECK(p != NULL && n > 0u && first > 1u && first < UINTPTR_MAX - n - 1u);
  CHECK(storage_aliases_live_reference(p, n) == 1);
  CHECK(storage_aliases_live(p, n) == 1);
  /* These boundary/interior probes must hit this allocation irrespective of
   * allocator placement or neighboring allocations. */
  CHECK(storage_aliases_live((void *)first, 1u) == 1);
  CHECK(storage_aliases_live((void *)(first + n - 1u), 1u) == 1);
  CHECK(storage_aliases_live((void *)(first + n / 2u), 1u) == 1);
  CHECK(storage_aliases_live((void *)(first - 1u), 2u) == 1);
  CHECK(storage_aliases_live((void *)(first + n - 1u), 2u) == 1);
  CHECK(storage_aliases_live((void *)(first - 1u), n + 2u) == 1);
  differential((void *)(first - 1u), 1u);
  differential((void *)(first + n), 1u);
  differential(p, 0u);
  /* Every enumerated protected span must be inside the conservative envelope.
   * Direct positive/reference checks above verify the explicit class inventory;
   * an encompassing neighbor can mask a missing recorder call, so this check
   * does not replace review of the single f32_calloc allocation boundary. */
  CHECK(f32_allocation_envelope.initialized && !f32_allocation_envelope.disabled);
  CHECK(f32_allocation_envelope.low <= first);
  CHECK(f32_allocation_envelope.high >= first + n);
}

static void live_span(enum allocation_class kind, const void *p, size_t n) {
  CHECK(kind < LIVE_CLASS_COUNT);
  protected_span(p, n);
  covered |= 1u << kind;
}

static void candidates(void) {
  int stack_value = 0;
  differential(NULL, 0u);
  differential(NULL, 1u);
  differential((void *)(UINTPTR_MAX - 1u), 4u);
  differential((void *)UINTPTR_MAX, 0u);
  differential(&stack_value, sizeof(stack_value));
  if (f32_allocation_envelope.initialized) {
    uintptr_t lo = f32_allocation_envelope.low;
    uintptr_t hi = f32_allocation_envelope.high;
    CHECK(lo > 1u && hi < UINTPTR_MAX);
    differential((void *)(lo - 1u), 1u); /* exact low boundary contact */
    differential((void *)hi, 1u);        /* exact high boundary contact */
    differential((void *)1u, 1u);
    differential((void *)(UINTPTR_MAX - 2u), 1u);
    differential((void *)(lo - 1u), 2u);
    differential((void *)(hi - 1u), 2u);
    differential((void *)(lo - 1u), hi - lo + 2u);
    for (uintptr_t k = 0u; k <= 64u; ++k) {
      uintptr_t point = lo + (hi - lo) / 64u * k;
      differential((void *)point, 1u);
      differential((void *)point, 7u);
    }
  }
}

static et_f32_tensor *tensor(size_t rank, const uint64_t *shape) {
  et_f32_tensor *result = NULL;
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_create_v1(rank, shape, &result, &error) == 0);
  CHECK(result != NULL);
  return result;
}

static void ignored_extents(void) {
  f32_allocation_envelope_state before = f32_allocation_envelope;
  f32_record_allocation(NULL, SIZE_MAX, SIZE_MAX);
  f32_record_allocation((void *)1u, 0u, SIZE_MAX);
  f32_record_allocation((void *)1u, SIZE_MAX, 0u);
  size_t count_before = successful_allocations;
  void *zero = f32_calloc(0u, 4u);
  CHECK(successful_allocations == count_before + (zero != NULL));
  free(zero);
  CHECK(f32_allocation_envelope.low == before.low);
  CHECK(f32_allocation_envelope.high == before.high);
  CHECK(f32_allocation_envelope.initialized == before.initialized);
  CHECK(f32_allocation_envelope.disabled == before.disabled);
}

static void empty_and_history(void) {
  CHECK(live_tensors == NULL && retired_tensors == NULL);
  CHECK(!f32_allocation_envelope.initialized);
  CHECK(storage_aliases_live(NULL, 1u) == 1);
  CHECK(storage_aliases_live((void *)(UINTPTR_MAX - 1u), 4u) == 1);
  candidates();
  /* A constructor failure after one successful allocation frees its storage,
   * but must not erase address history or change the old empty-scan result. */
  et_f32_tensor *failed = NULL;
  et_f32_tensor_error error;
  const uint64_t shape[] = {3u, 5u};
  et_f32_tensor_test_fail_alloc_after_v1(1u);
  CHECK(et_f32_tensor_create_v1(2u, shape, &failed, &error) != 0);
  CHECK(failed == NULL && successful_allocations == 1u);
  CHECK(live_tensors == NULL && retired_tensors == NULL);
  CHECK(f32_allocation_envelope.initialized);
  uintptr_t old_low = f32_allocation_envelope.low;
  uintptr_t old_high = f32_allocation_envelope.high;
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(f32_allocation_envelope.low == old_low);
  CHECK(f32_allocation_envelope.high == old_high);
  CHECK(!f32_allocation_envelope.disabled);
  CHECK(storage_aliases_live(NULL, 1u) == 1);
  CHECK(storage_aliases_live((void *)(UINTPTR_MAX - 1u), 4u) == 1);
  candidates();
  /* Recorded but unregistered storage is a deliberate hole. The envelope is
   * only a negative filter, never an assertion that everything inside aliases. */
  void *hole = f32_calloc(1u, 4096u);
  CHECK(hole != NULL);
  uintptr_t address = (uintptr_t)hole;
  CHECK(f32_allocation_envelope.low <= address);
  CHECK(f32_allocation_envelope.high >= address + 4096u);
  CHECK(storage_aliases_live(hole, 4096u) == 0);
  differential(hole, 4096u);
  free(hole);
  CHECK(storage_aliases_live((void *)address, 4096u) == 0);
  differential((void *)address, 4096u);
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(f32_allocation_envelope.low <= old_low);
  CHECK(f32_allocation_envelope.high >= old_high);
}

static void live_and_retired(void) {
  const uint64_t shape[] = {3u, 5u};
  const uint64_t empty_shape[] = {0u};
  et_f32_tensor_error error;
  et_f32_tensor *a = tensor(2u, shape), *b = tensor(2u, shape);
  et_f32_tensor *scalar = tensor(0u, NULL), *empty = tensor(1u, empty_shape);
  et_f32_tensor *borrow_owner = tensor(2u, shape);
  et_f32_tensor *gradient_source = tensor(2u, shape);
  et_f32_tensor_borrow *borrow = NULL;
  CHECK(et_f32_tensor_borrow_begin_v1(borrow_owner, &borrow, &error) == 0);
  et_f32_tensor_copy_assignment_v1 assignment = {sizeof(assignment), a, b};
  et_f32_tensor_copy_plan *copy = NULL;
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &assignment, &copy, &error) == 0);
  et_f32_parameter *parameter = NULL, *reset_parameter = NULL;
  CHECK(et_f32_parameter_create_v1(b, &parameter, &error) == 0);
  CHECK(et_f32_parameter_create_v1(b, &reset_parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(reset_parameter, &reset_parameter, &error) == 0);
  et_f32_gradient_contribution_v1 contribution = {sizeof(contribution), parameter, gradient_source, 0u};
  et_f32_gradient_plan *gradient = NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &contribution, UINT32_C(0x3f800000), &gradient, &error) == 0);
  et_f32_gradient_reset_plan *reset = NULL;
  CHECK(et_f32_gradient_reset_plan_prepare_v1(1u, &reset_parameter, &reset, &error) == 0);

  live_span(TENSOR_CONTROL, a, sizeof(*a));
  live_span(TENSOR_SHAPE, a->shape, a->rank * sizeof(*a->shape));
  live_span(TENSOR_STRIDES, a->strides, a->rank * sizeof(*a->strides));
  live_span(TENSOR_PAYLOAD, a->data, a->byte_length);
  live_span(BORROW_CONTROL, borrow, sizeof(*borrow));
  live_span(COPY_CONTROL, copy, sizeof(*copy));
  live_span(COPY_ASSIGNMENTS, copy->assignments, copy->count * sizeof(*copy->assignments));
  live_span(PARAMETER_CONTROL, parameter, sizeof(*parameter));
  live_span(GRADIENT_CONTROL, gradient, sizeof(*gradient));
  live_span(GRADIENT_ENTRIES, gradient->entries, gradient->count * sizeof(*gradient->entries));
  live_span(GRADIENT_PREPARED, gradient->entries[0].prepared, parameter->gradient->byte_length);
  live_span(RESET_CONTROL, reset, sizeof(*reset));
  live_span(RESET_PARAMETERS, reset->parameters, reset->count * sizeof(*reset->parameters));
  CHECK(LIVE_CLASS_COUNT == 13);
  CHECK(covered == (1u << 13u) - 1u);
  CHECK(scalar->rank == 0u && scalar->byte_length == 4u);
  protected_span(scalar->data, sizeof(float));
  CHECK(empty->byte_length == 0u && empty->data == NULL);
  differential(empty->data, 0u);
  protected_span(empty, sizeof(*empty));
  candidates();
  CHECK(storage_aliases_live(NULL, 1u) == 1);
  CHECK(storage_aliases_live((void *)(UINTPTR_MAX - 1u), 4u) == 1);

  const void *retired[6] = {a, borrow, copy, parameter, gradient, reset};
  const size_t sizes[6] = {sizeof(*a), sizeof(*borrow), sizeof(*copy), sizeof(*parameter), sizeof(*gradient), sizeof(*reset)};
  CHECK(et_f32_gradient_plan_release_v1(&gradient, &error) == 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_release_v1(&copy, &error) == 0);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&reset_parameter, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&a, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&b, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&borrow_owner, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&gradient_source, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&scalar, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&empty, &error) == 0);
  CHECK(!live_tensors && !live_borrows && !live_parameters && !live_copy_plans && !live_gradient_plans && !live_reset_plans);
  CHECK(retired_tensors && retired_borrows && retired_parameters && retired_copy_plans && retired_gradient_plans && retired_reset_plans);
  for (size_t i = 0u; i < 6u; ++i) protected_span(retired[i], sizes[i]);
  candidates();
}

static void disablement(int endpoint) {
  CHECK(!f32_allocation_envelope.disabled);
  if (endpoint) f32_record_allocation((void *)(UINTPTR_MAX - 1u), 1u, 2u);
  else f32_record_allocation((void *)1u, SIZE_MAX, 2u);
  CHECK(f32_allocation_envelope.disabled);
  if (!live_tensors && !retired_tensors) {
    CHECK(!f32_allocation_envelope.initialized);
    CHECK(storage_aliases_live(NULL, 1u) == 1);
    CHECK(storage_aliases_live((void *)(UINTPTR_MAX - 1u), 4u) == 1);
  }
  uintptr_t low = f32_allocation_envelope.low, high = f32_allocation_envelope.high;
  f32_record_allocation((void *)1u, 1u, 1u);
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(f32_allocation_envelope.disabled);
  CHECK(f32_allocation_envelope.low == low && f32_allocation_envelope.high == high);
  et_f32_tensor *scalar = tensor(0u, NULL);
  CHECK(f32_allocation_envelope.disabled);
  differential(scalar, sizeof(*scalar));
  differential(scalar->data, scalar->byte_length);
  differential(NULL, 1u);
  differential((void *)1u, 1u);
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(&scalar, &error) == 0);
  CHECK(f32_allocation_envelope.disabled);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  ignored_extents();
  if (strcmp(argv[1], "uninitialized") == 0) {
    disablement(0);
    printf("I2 alias envelope PASS: %zu checks; uninitialized disablement\n", checks);
    return 0;
  }
  empty_and_history();
  live_and_retired();
  ignored_extents();
  if (strcmp(argv[1], "product") == 0) disablement(0);
  else if (strcmp(argv[1], "endpoint") == 0) disablement(1);
  else CHECK(strcmp(argv[1], "coverage") == 0);
  printf("I2 alias envelope PASS: %zu checks; 13 live classes; 6 retired kinds; %s\n", checks, argv[1]);
  return 0;
}
