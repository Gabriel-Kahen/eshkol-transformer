#define T1_STRINGIFY_INNER(value) #value
#define T1_STRINGIFY(value) T1_STRINGIFY_INNER(value)

#ifndef T1_PRIVATE_LINK_NAME
#ifndef T1_PRIVATE_SYMBOL
#error "T1_PRIVATE_SYMBOL or T1_PRIVATE_LINK_NAME must name the probe symbol"
#endif
#define T1_PRIVATE_LINK_NAME T1_STRINGIFY(T1_PRIVATE_SYMBOL)
#endif

extern void t1_private_candidate(void) __asm__(T1_PRIVATE_LINK_NAME);

int t1_private_probe_call(void) {
  t1_private_candidate();
  return 0;
}
