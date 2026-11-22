/* Development witness using an explicitly selected, already installed GGML. */
#include "gguf.h"

#include <stdio.h>
#include <string.h>

static int put_u64(FILE *out, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, sizeof bytes, out) == sizeof bytes;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    struct ggml_context *tensors = NULL;
    struct gguf_context *file = gguf_init_from_file(
        argv[1], (struct gguf_init_params){.no_alloc = false, .ctx = &tensors});
    struct gguf_context *writer = NULL;
    FILE *out = NULL;
    int ok = file && tensors;
    if (!ok) goto done;
    writer = gguf_init_empty();
    if (!writer) { ok = 0; goto done; }
    gguf_set_kv(writer, file);
    const int64_t count = gguf_get_n_tensors(file);
    ok = gguf_get_version(file) == 3 && gguf_get_alignment(file) == 32
        && count >= 0 && count <= 4096;
    if (!ok) goto done;
    out = fopen(argv[3], "wb");
    ok = out && fwrite("IO5GGUF1", 1, 8, out) == 8
        && put_u64(out, (uint64_t)count);
    for (int64_t i = 0; ok && i < count; ++i) {
        const char *name = gguf_get_tensor_name(file, i);
        const struct ggml_tensor *tensor = ggml_get_tensor(tensors, name);
        const int64_t *shape = gguf_get_tensor_ne(file, i);
        const size_t name_bytes = strlen(name);
        ok = tensor && tensor->data && name_bytes > 0 && name_bytes < 64
            && gguf_get_tensor_type(file, i) == GGML_TYPE_F32
            && tensor->type == GGML_TYPE_F32
            && ggml_nbytes(tensor) == gguf_get_tensor_size(file, i)
            && ggml_nbytes(tensor) <= 32 * 1024 * 1024;
        if (!ok) break;
        ok = put_u64(out, name_bytes) && fwrite(name, 1, name_bytes, out) == name_bytes;
        for (int dim = 0; ok && dim < GGML_MAX_DIMS; ++dim) {
            ok = shape[dim] > 0 && shape[dim] == tensor->ne[dim]
                && put_u64(out, (uint64_t)shape[dim]);
        }
        const size_t bytes = ggml_nbytes(tensor);
        ok = ok && put_u64(out, bytes) && fwrite(tensor->data, 1, bytes, out) == bytes;
        /* Populate a writer from the reader's allocated, live GGML tensors. */
        if (ok) gguf_add_tensor(writer, tensor);
    }
    if (out) {
        if (fclose(out) != 0) ok = 0;
        out = NULL;
    }
    if (ok) ok = gguf_write_to_file(writer, argv[2], false);
done:
    if (out) fclose(out);
    if (writer) gguf_free(writer);
    if (file) gguf_free(file);
    if (tensors) ggml_free(tensors);
    if (!ok) fputs("IO5 native GGUF witness failed\n", stderr);
    return ok ? 0 : 1;
}
