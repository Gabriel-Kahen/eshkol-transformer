"""Direct normal/repeat FILE witness; launched only by the reviewed locked recipe."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster, validate
from tests.d2.test_resources import write_d1_resource

OVERLAY = ROOT / 'tests/cpu4_checkpoint_factory/file_native_overlay.json'
CALLER = ROOT / 'tests/cpu4_checkpoint_factory/file_runtime.esk'
FINGERPRINT = 'sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704'
TEST_FLAGS = {
    'src/eshkol_transformer/m3_call_f32_integration.c': ['ET_F32_TENSOR_TESTING'],
    'native/o2_optimizer.c': ['ET_O2_TESTING'],
    'native/c2_checkpoint_reader.c': ['ET_C2_CHECKPOINT_READER_TESTING'],
    'native/c2_checkpoint_core.c': ['ET_C2_CHECKPOINT_CORE_TESTING'],
    'native/checkpoint_io.c': ['ET_CHECKPOINT_IO_TESTING'],
}
OBSERVER = {'source': 'tests/cpu4_candidate_children/native_observer.c',
            'defines': ['ET_O2_TESTING', 'ET_F32_TENSOR_TESTING']}


def plan(roster, overlay):
    validate(ROOT, roster)
    assert overlay == {'schema': 'cpu4-factory-file-test-overlay-v1', 'mode': 'normal',
                       'per_tu_defines': TEST_FLAGS, 'additional_translation_units': [OBSERVER]}, 'test overlay mismatch'
    units = copy.deepcopy(roster['translation_units'])
    for unit in units:
        unit['defines'] += overlay['per_tu_defines'].get(unit['source'], [])
    units += copy.deepcopy(overlay['additional_translation_units'])
    assert len({unit['source'] for unit in units}) == len(units), 'duplicate test TU'
    return units


def hashes(directory, target):
    rows = []
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path != target:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1048576), b''):
                    digest.update(chunk)
            rows.append(digest.hexdigest() + '  ' + path.relative_to(directory).as_posix() + '\n')
    target.write_text(''.join(rows))


def dependencies(depfile):
    text = depfile.read_text().replace('\\\n', ' ')
    return [Path(name).resolve() for name in shlex.split(text.split(':', 1)[1])]


def inventory_depfiles(out):
    # Discovered observations only. Never use this inventory as an allowlist.
    all_paths = set()
    inventory = {}
    for depfile in sorted(out.rglob('*.d')):
        paths = dependencies(depfile)
        inventory[depfile.relative_to(out).as_posix()] = [str(path) for path in paths]
        all_paths.update(paths)
    (out / 'observed-dependencies.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (out / 'observed-dependencies.sha256').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path) + '\n'
        for path in sorted(all_paths)))


def gate(out, runner, cc, cxx, compiler_source):
    assert out.is_absolute() and not out.exists() and not out.is_symlink(), 'fresh absolute output required'
    out.mkdir(mode=0o700)
    program_status, custody_status = 1, 0
    def run(label, argv, env=None):
        (out / (label + '.argv.json')).write_text(json.dumps([str(arg) for arg in argv]) + '\n')
        with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        (out / (label + '.exit')).write_text(str(result.returncode) + '\n')
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv)
    before = subprocess.check_output(['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'], cwd=ROOT)
    try:
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT)
        (out / 'source.before').write_bytes(before)
        roster = load_roster()
        overlay = json.loads(OVERLAY.read_text())
        units = plan(roster, overlay)
        (out / 'production-inputs.json').write_text(json.dumps(roster, indent=2) + '\n')
        (out / 'test-overlay.json').write_text(json.dumps(overlay, indent=2) + '\n')
        (out / 'selected-native-inputs.json').write_text(json.dumps(units, indent=2) + '\n')
        (out / 'profile.txt').write_text('candidate=ON\ncompile_jobs=1\nmode=normal\nSAN=NOT RUN\nclosure=discovery, independent review required\n')
        resources = out / 'resources'
        write_d1_resource(resources, ((3,197,3,41,7,2,9,5,1),), fingerprint=FINGERPRINT, vocab=256)
        hashes(resources, out / 'resource-bytes.sha256')
        objects = out / 'objects'
        objects.mkdir()
        cache = out / 'cache'
        cache.mkdir()
        flags = roster['cflags'] + ['-O2']
        for directory in roster['native_include_dirs']:
            flags += ['-I', str(ROOT / directory)]
        flags += ['-I', str(compiler_source / 'inc')]
        flags += ['-D' + feature for feature in roster['common_defines']]
        object_paths = []
        for index, unit in enumerate(units):
            stem = f'{index:02d}-' + unit['source'].replace('/', '_')
            target = objects / (stem + '.o')
            run('native.' + stem, ['timeout', '--foreground', '--signal=TERM', '--kill-after=5s', '120s', cc,
                *flags, *['-D' + feature for feature in unit['defines']], '-MMD', '-MF', objects / (stem + '.d'),
                '-c', ROOT / unit['source'], '-o', target])
            object_paths.append(target)
        library = out / 'libcpu4_factory_file.a'
        run('archive', ['ar', 'rcsD', library, *object_paths])
        run('archive.members', ['ar', 't', library])
        run('archive.symbols', ['nm', '-a', '--format=posix', library])
        run('archive.imports', ['nm', '-u', '--format=posix', library])
        (out / 'cxx-link').write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(str(cxx)) + ' "$@"\n')
        (out / 'cxx-link').chmod(0o500)
        environment = dict(os.environ)
        for name in ('ESHKOL_PATH', 'ESHKOL_JIT_CACHE_DIR'):
            environment.pop(name, None)
        environment.update(ESHKOL_JIT_CACHE='0', XDG_CACHE_HOME=str(cache),
                           ESHKOL_CXX_COMPILER=str(out / 'cxx-link'), ESHKOL_LIB_DIR=str(ROOT / 'lib'))
        includes = [arg for directory in roster['source_include_dirs'] for arg in ('-I', str(ROOT / directory))]
        raw = out / 'runtime.raw.o'
        run('compile', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '1200s', runner,
            '--strict-types', '--no-stdlib', '--optimize', '0', *includes, '-L', out, '--lib', 'cpu4_factory_file',
            '--emit-object', '--dump-ir', '--emit-depfile', out / 'runtime.d', CALLER, '-o', raw], environment)
        for suffix in ('', '.ll', '.bc'):
            assert Path(str(raw) + suffix).stat().st_size > 0, 'compiler artifact missing'
        assert (out / 'runtime.d').stat().st_size > 0
        run('object.imports', ['nm', '-u', '--format=posix', raw])
        run('object.symbols', ['nm', '-a', '--format=posix', raw])
        executable = out / 'runtime'
        run('link', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '120s', runner,
            '--no-stdlib', '-L', out, '--lib', 'cpu4_factory_file', raw, '-o', executable], environment)
        run('elf.header', ['readelf', '-h', executable])
        run('elf.notes', ['readelf', '-n', executable])
        run('elf.symbols', ['nm', '-a', '--format=posix', executable])
        run('elf.imports', ['nm', '-u', '--format=posix', executable])
        for repeat in (1, 2):
            environment['ESHKOL_ARENA_POISON'] = '1'
            label = f'run{repeat}'
            run(label, ['timeout', '--foreground', '--signal=TERM', '--kill-after=5s', '120s',
                        executable, resources, out / f'saved{repeat}.c2'], environment)
            assert not (out / (label + '.stderr')).read_bytes(), 'runtime diagnostics'
            text = (out / (label + '.stdout')).read_text()
            assert 'CPU4 artificial coherent FILE factory data movement PASS: checks=' in text
        assert (out / 'run1.stdout').read_bytes() == (out / 'run2.stdout').read_bytes(), 'repeat output differs'
        assert (out / 'saved1.c2').read_bytes() == (out / 'saved2.c2').read_bytes(), 'repeat checkpoint bytes differ'
        program_status = 0
    except subprocess.CalledProcessError as error:
        program_status = error.returncode
    except Exception as error:
        (out / 'driver-error.txt').write_text(repr(error) + '\n')
    finally:
        try:
            inventory_depfiles(out)
            after = subprocess.check_output(['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'], cwd=ROOT)
            (out / 'source.after').write_bytes(after)
            assert before == after
            assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT)
        except Exception as error:
            custody_status = 74
            (out / 'custody-error.txt').write_text(repr(error) + '\n')
        (out / 'driver-program.exit').write_text(str(program_status) + '\n')
        (out / 'driver-custody.exit').write_text(str(custody_status) + '\n')
        hashes(out, out / 'SHA256SUMS')
    return custody_status or program_status


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit('file_driver OUTPUT RUNNER CC CXX COMPILER_SOURCE')
    raise SystemExit(gate(*map(Path, sys.argv[1:])))
