/* Private test observer only: the included production TU owns the real registry.
 * No observer, counter, or corruption helper is added to production authority. */
#include "../../native/e3_d2_native.c"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>

static int observing;
static size_t checks;
static size_t allocation_calls, free_calls, io_calls;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);
int __real_open(const char *, int, ...);
ssize_t __real_pread(int, void *, size_t, off_t);
int __real_close(int);

void *__wrap_malloc(size_t size) {
  assert(!observing); allocation_calls++; return __real_malloc(size);
}
void *__wrap_calloc(size_t count, size_t size) {
  assert(!observing); allocation_calls++; return __real_calloc(count, size);
}
void *__wrap_realloc(void *ptr, size_t size) {
  assert(!observing); allocation_calls++; return __real_realloc(ptr, size);
}
void __wrap_free(void *ptr) {
  assert(!observing); free_calls++; __real_free(ptr);
}
int __wrap_open(const char *path, int flags, ...) {
  assert(!observing); io_calls++;
  if ((flags & O_CREAT) != 0) {
    va_list args;
    va_start(args, flags);
    mode_t mode = (mode_t)va_arg(args, int);
    va_end(args);
    return __real_open(path, flags, mode);
  }
  return __real_open(path, flags);
}
ssize_t __wrap_pread(int fd, void *buf, size_t count, off_t offset) {
  assert(!observing); io_calls++; return __real_pread(fd, buf, count, offset);
}
int __wrap_close(int fd) {
  assert(!observing); io_calls++; return __real_close(fd);
}

/* Fixed observer capacity is test-only and covers every allocation owned by the
 * two real [1,2] datasets used here, including opaque I1 controls/leases. */
typedef struct snapshot {
  unsigned char bytes[8192];
  size_t used;
} snapshot;

static void save(snapshot *out, const void *data, size_t bytes) {
  assert(bytes <= sizeof(out->bytes) - out->used);
  if (bytes != 0u) memcpy(out->bytes + out->used, data, bytes);
  out->used += bytes;
}
#define SAVE(out, value) save((out), &(value), sizeof(value))

static void capture(snapshot *out) {
  memset(out, 0, sizeof(*out));
  SAVE(out, live_datasets);
  SAVE(out, dataset_shell_factory); SAVE(out, batch_shell_factory);
  SAVE(out, fail_stage); SAVE(out, read_fail_stage);
  SAVE(out, owned_allocations);
  SAVE(out, exact_read_fd_live); SAVE(out, exact_read_fd_peak);
  SAVE(out, allocation_calls); SAVE(out, free_calls); SAVE(out, io_calls);
  for (const et_d2_dataset_control *ds = live_datasets; ds;
       ds = ds->registry_next) {
    save(out, ds, sizeof(*ds));
    save(out, ds->shuffle_window, ds->shuffle_slots * sizeof(int64_t));
    const et_d2_batch_control *b = ds->current_batch;
    if (!b) continue;
    save(out, b, sizeof(*b));
    const et_i64_tensor *tensors[] = {b->input_tensor, b->target_tensor};
    const et_i64_tensor_borrow *borrows[] = {
        b->input_storage_borrow, b->target_storage_borrow};
    const et_kernel_tensor_view_v1 *views[] = {b->input_view, b->target_view};
    for (size_t i = 0; i < 2; ++i) {
      save(out, tensors[i], et_i64_tensor_test_tensor_control_bytes_v1());
      save(out, borrows[i], et_i64_tensor_test_borrow_bytes_v1());
      save(out, views[i], sizeof(*views[i]));
      save(out, views[i]->shape, 2u * sizeof(uint64_t));
      save(out, et_i64_tensor_test_stride_storage_v1(tensors[i]),
           2u * sizeof(size_t));
      save(out, views[i]->data, views[i]->byte_length);
    }
    save(out, b->mask_data, b->elements);
  }
}

static void probe(const void *owner, int64_t expected) {
  snapshot before, after;
  /* Every prior status, including a value outside the return table, is replaced. */
  for (int64_t prior = 0; prior < 8; ++prior) {
    last_status = prior;
    capture(&before);
    errno = EDOM;
    observing = 1;
    int64_t result = et_e3_d2_dataset_idle_preflight_v1(owner);
    observing = 0;
    assert(result == expected && et_d2_batch_last_status_v1() == expected);
    assert(errno == EDOM);
    capture(&after);
    assert(memcmp(&before, &after, sizeof(before)) == 0);
    checks++;
  }
}

static int64_t create(const void *owner) {
  int64_t generation = et_d2_batch_create_v1(owner, 1, 2, 34);
  assert(generation > 0);
  probe(owner, ET_D2_NATIVE_STATUS_INVALID_STATE); /* unpublished, unwritten */
  return generation;
}

static void seal(const void *owner, int64_t generation) {
  struct { int64_t length; unsigned char bytes[16]; } inputs = {16, {1}},
      targets = {16, {2}};
  inputs.bytes[8] = 3; targets.bytes[8] = 4;
  assert(et_d2_batch_write_pair_i64le_span_v1(
      owner, generation, 0, &inputs, 16, &targets, 16, 2) == 0);
  probe(owner, 2); /* written but unsealed */
  assert(et_d2_batch_seal_v1(owner, generation) == 0);
  probe(owner, 2);
}

int main(void) {
  unsigned char owners[3] = {11, 22, 33};
  probe(NULL, 1); probe(&owners[0], 1);
  probe((const void *)(uintptr_t)1, 1);
  probe((const void *)(uintptr_t)UINTPTR_MAX, 1);
  /* Real unreadable owner proves comparison-only on both unknown and enrolled
   * paths. Mapping is a test fixture, outside every observed call. */
  long page_bytes = sysconf(_SC_PAGESIZE);
  assert(page_bytes > 0);
  int fd = open("/dev/zero", O_RDONLY);
  assert(fd >= 0);
  void *opaque = mmap(NULL, (size_t)page_bytes, PROT_NONE, MAP_PRIVATE, fd, 0);
  assert(opaque != MAP_FAILED && close(fd) == 0);
  probe(opaque, 1);
  assert(et_d2_dataset_open_v1(opaque, 0) == 0);
  assert(et_d2_dataset_open_v1(&owners[0], 3) == 0);
  for (int64_t i = 0; i < 3; ++i)
    assert(et_d2_shuffle_window_store_v1(&owners[0], i, 9 - i) == 0);
  probe(opaque, 0); probe(&owners[0], 0); probe(&owners[1], 1);
  probe(&owners[0] + 1, 1); /* adjacent/copy is no enrollment */
  int64_t g = create(&owners[0]);
  probe(opaque, 0); /* busy other dataset does not affect idle owner */
  assert(et_d2_batch_release_v1(&owners[0], g) == 0);
  probe(&owners[0], 0); /* unpublished rollback */
  g = create(&owners[0]);
  seal(&owners[0], g);
  for (int i = 0; i < 3; ++i) {
    int64_t lease = et_d2_batch_borrow_begin_v1(&owners[0], g);
    assert(lease > 0);
    const et_kernel_tensor_view_v1 *in =
        et_d2_batch_borrow_inputs_v1(&owners[0], g, lease);
    const et_kernel_tensor_view_v1 *target =
        et_d2_batch_borrow_targets_v1(&owners[0], g, lease);
    const et_kernel_tensor_view_v1 *mask =
        et_d2_batch_borrow_loss_mask_v1(&owners[0], g, lease);
    assert(in && target && mask);
    assert(((const int64_t *)in->data)[0] == 1);
    assert(((const int64_t *)target->data)[1] == 4);
    assert(((const uint8_t *)mask->data)[1] == 1);
    probe(&owners[0], 2); probe(opaque, 0);
    assert(et_d2_batch_release_preflight_v1(&owners[0], g) == 2);
    assert(et_d2_dataset_close_v1(&owners[0]) == 2);
    probe(&owners[0], 2);
    assert(et_d2_batch_borrow_end_v1(&owners[0], g, lease) == 0);
    probe(&owners[0], 2);
  }
  assert(et_d2_batch_release_preflight_v1(&owners[0], g) == 0);
  assert(et_d2_batch_release_v1(&owners[0], g) == 0);
  probe(&owners[0], 0);
  assert(et_d2_batch_release_preflight_v1(&owners[0], g) == 1);
  probe(&owners[0], 0); /* stale generation has no idle authority */
  /* Every existing real partial-allocation rollback leaves idle enrollment. */
  for (int64_t stage = 1; stage <= ET_D2_TEST_FAIL_BEFORE_PUBLISH; ++stage) {
    et_d2_batch_test_fail_stage_v1(stage);
    assert(et_d2_batch_create_v1(&owners[0], 1, 2, 34) == 0);
    assert(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    probe(&owners[0], 0);
  }
  et_d2_batch_test_fail_stage_v1(0);
  g = create(&owners[0]); seal(&owners[0], g);
  /* Existing private exhaustion hook changes counters only in test setup. */
  et_d2_batch_test_exhaust_generations_v1();
  probe(&owners[0], 2); probe(opaque, 0);
  assert(et_d2_batch_borrow_begin_v1(&owners[0], g) == 0);
  assert(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_UNSUPPORTED);
  probe(&owners[0], 2);
  assert(et_d2_batch_release_v1(&owners[0], g) == 0);
  probe(&owners[0], 0);
  assert(et_d2_batch_create_v1(&owners[0], 1, 2, 34) == 0);
  assert(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_UNSUPPORTED);
  probe(&owners[0], 0);
  assert(et_d2_dataset_close_v1(&owners[0]) == 0);
  probe(&owners[0], 1); probe(opaque, 0);
  assert(et_d2_dataset_close_v1(opaque) == 0);
  probe(opaque, 1);
  assert(munmap(opaque, (size_t)page_bytes) == 0);
  /* Close also destroys a sealed current batch; reopen is real new enrollment. */
  assert(et_d2_dataset_open_v1(&owners[0], 0) == 0);
  g = create(&owners[0]); seal(&owners[0], g);
  assert(et_d2_dataset_close_v1(&owners[0]) == 0);
  probe(&owners[0], 1);
  assert(memcmp(owners, (unsigned char[]){11, 22, 33}, sizeof(owners)) == 0);
  assert(live_datasets == NULL && owned_allocations == 0);
  assert(et_d2_batch_test_live_count_v1() == 0);
  assert(et_d2_batch_test_borrow_count_v1() == 0);
  assert(exact_read_fd_live == 0);
  printf("E3-D2 native idle PASS: %zu exact status/resource snapshots\n", checks);
  return 0;
}
