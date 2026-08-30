extern _Thread_local volatile int attacker_tls;

void et_e1b_public_fixture_attacker_v1(void) {
  (void)attacker_tls;
}
