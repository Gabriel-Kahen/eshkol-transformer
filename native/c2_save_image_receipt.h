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
uint64_t et_c2_private_save_image_live_bytes_v1(void);
void et_c2_private_save_image_test_set_issued_v1(uint64_t);
void et_c2_private_save_image_test_corrupt_v1(int64_t, uint64_t);
void et_c2_private_save_image_test_writer_hook_v1(void (*)(int64_t));
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif
