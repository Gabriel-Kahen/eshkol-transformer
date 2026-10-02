"""Source checks for the isolated, noninstalled I2 envelope fault witness."""

from pathlib import Path
import runpy
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]


def check_wrap_boundary(builder: str, wrapper: str, driver: str) -> None:
    gate = 'E3_I2_POSTPREPARE_GUARD_LENGTH'
    start = builder.index(f'e3_i2_guard_length="${{{gate}:-}}"')
    exact_gate = builder[start:builder.index('\nfi\n', start)]
    if (builder.count(gate) != 1 or
            '"${e3_tuple_kind}" == zero' not in exact_gate or
            '"${package_policy}" == e3-private-aggregate' not in exact_gate):
        raise ValueError("wrapper must be admitted only by the exact zero tuple")
    partial = builder.index('"${e1b_cxx}" -r "${e3_zero_reloc_flags[@]}"')
    private = builder.index('"${e1b_tmp}/private.o"', partial)
    native = builder.index('"${package_native_objects[@]}"', private)
    if not partial < private < native:
        raise ValueError("interception must precede final object localization")
    for symbol in ('et_i2_private_copy_builder_prepare_v1',
                   'et_i2_private_copy_builder_abort_v1',
                   'et_i2_private_copy_builder_commit_v1',
                   'arena_allocate_vector_with_header',
                   'eshkol_get_raised_value'):
        if f'-Wl,--wrap={symbol}' not in builder or f'__real_{symbol}' not in wrapper:
            raise ValueError(f"missing real partial-link interception: {symbol}")
    prepare = wrapper.index('__real_et_i2_private_copy_builder_prepare_v1(builder);')
    armed = wrapper.index('prepared_builder = builder;', prepare)
    if not prepare < armed:
        raise ValueError("the real native prepare must precede fault arming")
    if ('builder == prepared_builder && consumed && !abort_count' not in wrapper or
            'status = __real_et_i2_private_copy_builder_abort_v1(builder);' not in wrapper or
            'require_witness(status == 0, "real native builder abort succeeded")' not in wrapper or
            'raised_condition == exception' not in wrapper):
        raise ValueError("abort and unchanged-condition assertions are required")
    if ('((string=? name "postprepare-envelope")' not in driver or
            '(i2-prepare-load\n' not in driver or
            '(census-before (e3-zero-negative-census model))' not in driver or
            'census-before (e3-zero-negative-census model)' not in driver or
            '(e3-zero-negative-sources sources)' not in driver or
            'source-bytes-before\n                    (e3-zero-negative-sources sources)' not in driver):
        raise ValueError("the fault case must observe authentic I2 state")


class I2PostprepareGuardContract(unittest.TestCase):
    def test_mutant_is_one_semantic_deletion_in_six_pinned_roots(self) -> None:
        module = runpy.run_path(str(ROOT / 'scripts/generate-e3-i2-guard-mutant.py'))
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / 'source'
            output.mkdir()
            records = module['stage'](ROOT, output)
            self.assertEqual(len(records), 6)
            self.assertEqual([row['staged'] for row in records],
                             [Path(relative).name for relative, _ in module['SOURCES']])
            for relative, _ in module['SOURCES'][1:5]:
                self.assertEqual((output / Path(relative).name).read_bytes(),
                                 (ROOT / relative).read_bytes())
            top_before, top_after = module['TOP_LOAD']
            self.assertEqual((output / 'zero_control_driver_root.esk').read_bytes().replace(
                top_after, top_before, 1),
                (ROOT / 'tests/e3_private/zero_control_driver_root.esk').read_bytes())
            abort_before, abort_after = module['ABORT_ARM']
            mutant = (output / 'i2_wave2_root.esk').read_bytes()
            self.assertEqual(mutant.replace(abort_after, abort_before, 1),
                             (ROOT / 'native/i2_wave2_root.esk').read_bytes())
            ordinary = (ROOT / 'tests/e3_private/zero_control_package_source_closure.txt').read_text().splitlines()
            closure = (ROOT / 'tests/e3_private/i2_guard_mutant_package_source_closure.txt').read_text().splitlines()
            self.assertEqual(closure[:6], [
                f'@BUILD@/source/i2-guard-overlay/{row["staged"]}' for row in records])
            self.assertEqual(closure[6:], ordinary[6:])
            self.assertEqual(closure.count('@BUILD@/source/i2-guard-overlay/i2_wave2_root.esk'), 1)
            self.assertNotIn('native/i2_wave2_root.esk', closure)
            normalize = '''
set -euo pipefail
PROJECT_ROOT=$1; e3_generated_source_root=$2; e3_i2_overlay_root=$3
die() { exit 41; }
raw_private_root="$PROJECT_ROOT/tests/e3_private/zero_control_driver_root.esk"
raw_package_bridge="$PROJECT_ROOT/tests/e3_private/zero_control_bridge.c"
raw_package_renames="$PROJECT_ROOT/tests/e3_private/zero_control_package_private_renames.txt"
raw_public_exports="$PROJECT_ROOT/tests/e3_private/zero_control_package_public_exports.txt"
raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib" "$PROJECT_ROOT/internal/c1/lib"
  "$PROJECT_ROOT/internal/t1/lib" "$PROJECT_ROOT/src"
  "$PROJECT_ROOT/internal/d2/lib" "$PROJECT_ROOT/internal/e3/lib")
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
printf '%s\\n' "$e3_i2_overlay_root/i2_wave2_root.esk" |
  e3_normalize_source_dependencies
'''
            mapped = subprocess.run(
                ['bash', '-c', normalize, 'normalize', str(ROOT), str(output.parent),
                 str(output)], capture_output=True, text=True, check=True)
            self.assertEqual(mapped.stdout.strip(), closure[5])

            forged = Path(temporary) / 'forged'
            for relative, _ in module['SOURCES']:
                target = forged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            (forged / 'native/i2_wave2_root.esk').write_bytes(
                (forged / 'native/i2_wave2_root.esk').read_bytes().replace(
                    abort_before, abort_after, 1))
            second = Path(temporary) / 'second'
            second.mkdir()
            with self.assertRaisesRegex(ValueError, 'SHA drifted'):
                module['stage'](forged, second)

    def test_partial_link_and_real_native_calls(self) -> None:
        builder = (ROOT / 'scripts/build-e1b-consumer.sh').read_text()
        wrapper = (ROOT / 'tests/e3_private/i2_postprepare_wrap.c').read_text()
        driver = (ROOT / 'tests/e3_private/zero_control_negative_driver.esk').read_text()
        check_wrap_boundary(builder, wrapper, driver)
        for changed in (
            builder.replace('"${e1b_cxx}" -r "${e3_zero_reloc_flags[@]}"',
                            '"${e1b_cxx}" -r', 1),
            builder.replace('-Wl,--wrap=et_i2_private_copy_builder_abort_v1',
                            '-Wl,--wrap=missing_abort', 1),
            builder.replace('"${e3_tuple_kind}" == zero ]] ||',
                            '"${e3_tuple_kind}" == base ]] ||', 1),
        ):
            with self.assertRaises(ValueError):
                check_wrap_boundary(changed, wrapper, driver)
        for changed in (
            wrapper.replace('__real_et_i2_private_copy_builder_prepare_v1(builder);',
                            '0;', 1),
            wrapper.replace('builder == prepared_builder && consumed && !abort_count',
                            'consumed && !abort_count', 1),
        ):
            with self.assertRaises(ValueError):
                check_wrap_boundary(builder, changed, driver)
        with self.assertRaises(ValueError):
            check_wrap_boundary(builder, wrapper, driver.replace(
                'census-before (e3-zero-negative-census model)',
                'census-before census-before', 1))


if __name__ == '__main__':
    unittest.main()
