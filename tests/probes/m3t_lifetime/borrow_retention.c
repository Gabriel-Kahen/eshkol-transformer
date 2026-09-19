/* Development-only measurement of merged I2's documented lease tombstones.
 * This includes the implementation only to count private retired shells. */
#include "../../../native/f32_tensor.c"

int main(int argc, char **argv) {
  et_f32_tensor *tensor = NULL;
  et_f32_tensor_error error;
  const uint64_t shape[] = {1};
  const size_t count = argc == 2 ? (size_t)strtoull(argv[1], NULL, 10) : 30000;
  size_t retired = 0;
  if (et_f32_tensor_create_v1(1, shape, &tensor, &error)) return 1;
  for (size_t i = 0; i < count; ++i) {
    et_f32_tensor_borrow *borrow = NULL;
    if (et_f32_tensor_borrow_begin_v1(tensor, &borrow, &error) ||
        et_f32_tensor_borrow_end_v1(&borrow, &error)) return 2;
  }
  for (const et_f32_tensor_borrow *p = retired_borrows; p; p = p->registry_next)
    ++retired;
  printf("iterations=%zu retired=%zu shell_bytes=%zu retained_bytes=%zu live=%s\n",
         count, retired, sizeof(et_f32_tensor_borrow),
         retired * sizeof(et_f32_tensor_borrow), live_borrows ? "nonzero" : "zero");
  if (retired != count || live_borrows) return 3;
  return et_f32_tensor_destroy_v1(&tensor, &error);
}
