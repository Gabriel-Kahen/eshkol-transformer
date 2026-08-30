#define D1_STRINGIFY_INNER(value) #value
#define D1_STRINGIFY(value) D1_STRINGIFY_INNER(value)

#ifndef D1_PRIVATE_LINK_NAME
#ifndef D1_PRIVATE_SYMBOL
#error "D1_PRIVATE_SYMBOL or D1_PRIVATE_LINK_NAME must name the link-probe symbol"
#endif
#define D1_PRIVATE_LINK_NAME D1_STRINGIFY(D1_PRIVATE_SYMBOL)
#endif

extern void d1_private_candidate(void) __asm__(D1_PRIVATE_LINK_NAME);

#ifdef D1_RELOCATABLE_PROBE
int d1_private_probe_call(void) {
  d1_private_candidate();
  return 0;
}
#else
int main(void) {
  d1_private_candidate();
  return 0;
}
#endif
