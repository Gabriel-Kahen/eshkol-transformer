#ifndef ET_C2_SAVE_IMAGE_RECEIPT_H
#define ET_C2_SAVE_IMAGE_RECEIPT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifdef ET_C2_SAVE_IMAGE_RECEIPT_PRIVATE
int64_t et_c2_private_save_image_adopt_v1(const void *, int64_t, void *);
int64_t et_c2_private_save_image_digest_v1(int64_t, void *);
int64_t et_c2_private_save_image_publish_v1(int64_t, const char *, int64_t);
int64_t et_c2_private_save_image_release_v1(int64_t);
#ifdef ET_C2_SAVE_IMAGE_RECEIPT_TESTING
enum {
  ET_C2_IMAGE_TEST_FAIL_ENTRY_ALLOC = 1,
  ET_C2_IMAGE_TEST_FAIL_VECTOR_ALLOC = 2,
  ET_C2_IMAGE_TEST_FAIL_AFTER_SHA = 3,
  ET_C2_IMAGE_TEST_FAIL_RELEASE = 4
};
uint64_t et_c2_private_save_image_live_bytes_v1(void);
uint64_t et_c2_private_save_image_live_count_v1(void);
void et_c2_private_save_image_test_fail_next_v1(int64_t);
void et_c2_private_save_image_test_set_issued_v1(uint64_t);
void et_c2_private_save_image_test_corrupt_v1(int64_t, uint64_t);
void et_c2_private_save_image_test_writer_hook_v1(void (*)(int64_t));
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif
