#include <dlfcn.h>
#include <stdio.h>

#include "tr3_c_private_initializer_bridge.h"

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 3;
  }
  int (*initialize)(void) = (int (*)(void))dlsym(
      library, "et_tr3_c_private_initialize_v1");
  if (!initialize) return 4;
  if (dlsym(library, "__eshkol_lib_init__")) return 11;
  if (dlsym(library, "get_global_arena_shared")) return 9;
  if (dlsym(library, "__repl_shared_arena")) return 10;
  if (dlsym(library, "tr3-lease-create-internal")) return 5;
  if (dlsym(library, "tr3-lease-unenroll-internal!")) return 14;
  if (dlsym(library, "tr3-c-trainer-load-state-internal!")) return 6;
  if (dlsym(library, "et_tr3_c_private_i2_restore_create_v1")) return 7;
  if (initialize() != ET_TR3_C_INIT_READY_V1) return 12;
  if (initialize() != ET_TR3_C_INIT_READY_V1) return 13;
  /* Initialized root objects may reference code in the package. Keep it loaded. */
  puts("TR3 linked private dynamic boundary PASS");
  return 0;
}
