/* Retain the two-entry CLI3 executable ABI; route dispatch to generation. */
#define cli3_private_dispatch_cabi_v1 cli3_generate_private_dispatch_cabi_v1
#include "cli3_package_bridge.c"
#undef cli3_private_dispatch_cabi_v1
