#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 3;
  }
  if (!dlsym(library, "__eshkol_lib_init__")) return 4;
  if (dlsym(library, "get_global_arena_shared")) return 9;
  if (dlsym(library, "__repl_shared_arena")) return 10;
  if (dlsym(library, "tr3-lease-create-internal")) return 5;
  if (dlsym(library, "tr3-c-trainer-load-state-internal!")) return 6;
  if (dlsym(library, "et_tr3_c_private_i2_restore_create_v1")) return 7;
  if (dlclose(library) != 0) return 8;
  puts("TR3 linked private dynamic boundary PASS");
  return 0;
}
