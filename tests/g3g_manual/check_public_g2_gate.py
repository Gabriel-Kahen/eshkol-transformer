"""Source admission for the recovered installed G2 AOT/SAN runner."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def check(builder: str, runner: str, caller: str, providers: str) -> None:
    required_builder = (
        'case "${G3G_MANUAL_SANITIZE:-0}" in',
        '"${package_policy}" == g3g-manual-public-aggregate &&\n       "${e3_private_test_sanitize}" == 0',
        'g3g_manual_test_sanitize=1',
        'g3g_manual_test_sanitize_flags=(-O1 -g -fsanitize=address,undefined',
        '"${g3g_manual_test_sanitize_flags[@]}" -c -x ir "${e1b_private_compile_ir}"',
        '"${g3g_manual_test_sanitize_flags[@]}" \\\n  "${package_bridge_flags[@]}"',
        'package_native_cflags+=("${g3g_manual_test_sanitize_flags[@]}")',
        '"${g3g_manual_test_reloc_flags[@]}" \\',
        'elif [[ "${g3g_manual_test_sanitize}" == 1 ]]; then',
        'G3G generated ASan IR helper source pins differ',
        'G3G generated ASan IR helper source is missing or symlinked',
        'G3G generated ASan IR exact streaming rewrite failed',
        'G3G generated private object lacks exact ASan IR evidence',
        'G3G installed sanitizer changed noninstrumentation imports',
        'e1b_compare_exact_undefined_allowlist \\\n    "${e1b_tmp}/expected-undefined.txt" "${e1b_tmp}/noninstrumentation-undefined.txt"',
        'cp "${e1b_tmp}/private.asan.json"',
        'cp "${e1b_tmp}/private.asan-imports.txt"',
    )
    required_runner = (
        'G3G_MANUAL_SANITIZE=0 /usr/bin/bash',
        'G3G_MANUAL_SANITIZE=1 /usr/bin/bash',
        'G2 evidence directory already exists',
        'scripts/test-g3g-manual-public.sh',
        'tests/g3g_manual/public_g2_runtime.esk',
        'check_public_closure.py',
        '-fsanitize=address,undefined -fno-omit-frame-pointer',
        'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
        'UBSAN_OPTIONS=halt_on_error=1',
        'G3G-MANUAL-PUBLIC-G2-PASS',
        'cmp "${source_pin}" "${source_after}"',
        'private.asan.ll private.asan.json private.asan-imports.txt',
        'noninstrumentation-undefined.txt sanitizer-undefined.txt',
        'normal) modes=(normal)', 'sanitize) modes=(sanitize)',
        'all) modes=(normal sanitize)',
        'G3G-MANUAL-PUBLIC-G2-GATE-PASS mode=%s',
        '"${runner}" --strict-types --no-stdlib',
        'g2_compile -O 0 -L "${directory}/installed"',
        '"${directory}/public-g2-runtime"',
    )
    for required in required_builder:
        if required not in builder:
            raise ValueError(f'installed SAN build contract missing: {required}')
    for required in required_runner:
        if required not in runner:
            raise ValueError(f'installed G2 run contract missing: {required}')
    if builder.count('case "${G3G_MANUAL_SANITIZE:-0}" in') != 1 or builder.count('g3g_manual_test_sanitize=1') != 1:
        raise ValueError('installed SAN selector duplicated')
    for required in ('g3g_manual_provider_mode=normal',
                     '"${g3g_manual_test_sanitize:-0}" == 1',
                     'g3g_manual_provider_mode=sanitize',
                     '"${e1b_tmp}/providers/${g3g_provider}" "${g3g_manual_provider_mode}"'):
        if required not in providers:
            raise ValueError('provider mode propagation missing')
    # No E3 selector is broadened to an installed tuple.
    if 'die "E3 generated ASan IR is restricted to private test tuples"' not in builder:
        raise ValueError('E3 private IR selector changed')
    if 'G3G_MANUAL_SANITIZE' in caller or re.search(r'\(load\b|g3c4-.*-private', caller):
        raise ValueError('installed caller imports test or private authority')
    if caller.count('(require transformer.') != 6 or caller.count('G3G-MANUAL-PUBLIC-G2-PASS') != 1:
        raise ValueError('installed caller surface changed')
    for required in ("':profile 'diagnostic-c4-p2g2", "':max-new-tokens 2",
                     '(policy #f)', 'first EOS generator closes before all output reads',
                     'two-token generator closes before all output reads',
                     '(rng-policy eos-rng)', 'dead RNG cannot become a constructor source'):
        if required not in caller:
            raise ValueError('installed G2 ownership witness missing')


def main(root: Path = ROOT) -> None:
    check((root / 'scripts/build-e1b-consumer.sh').read_text(),
          (root / 'scripts/test-g3g-manual-public-g2.sh').read_text(),
          (root / 'tests/g3g_manual/public_g2_runtime.esk').read_text(),
          (root / 'scripts/g3g-manual-native-inputs.sh').read_text())


if __name__ == '__main__':
    main()
    print('installed public G2 source gate: PASS')
