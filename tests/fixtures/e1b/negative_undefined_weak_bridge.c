extern void attacker_weak(void) __attribute__((weak));

void et_e1b_public_fixture_attacker_v1(void) {
  if (attacker_weak != 0) {
    attacker_weak();
  }
}
