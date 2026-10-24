"""SOURCE-ONLY staged normal/repeat driver; root owns composition/pins/launch."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster, manifest, validate
from tests.d2.test_resources import write_d1_resource

CALLER = ROOT / 'tests/cpu4_trained_g2/runtime.esk'
OVERLAY = ROOT / 'tests/cpu4_trained_g2/normal_native_overlay.json'
DEPENDENCIES = 'tests/cpu4_trained_g2/DEPENDENCIES.sha256'
FINGERPRINT = 'sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704'
TEST_FLAGS = {'native/o2_optimizer.c': ['ET_O2_TESTING'],
              'src/eshkol_transformer/m3_call_f32_integration.c': ['ET_F32_TENSOR_TESTING']}
OBSERVERS = [{'source': 'tests/cpu4_heldout/inspect.c', 'defines': ['ET_O2_TESTING']},
             {'source': 'tests/cpu4_candidate_children/native_observer.c',
              'defines': ['ET_O2_TESTING', 'ET_F32_TENSOR_TESTING']}]
MODES = ('two-token', 'first-eos', 'second-eos')
EXPECTED_B125_SHA256 = '499008516bce12c9ca07de19e97cee99ec373b0070cf13b34d035296889692cc'
# Mandatory root composition dependency; this isolated fd0 leaf is not executable.
REQUIRED_REPAIR = {
    'native/o2_optimizer.c': 'bc83b46a3a0b0c96dea06d81abed28baaf0655f5781b1a09635f77b41dee7f47',
    'native/tr3_c_o2_restore.inc': '5fd13e2c4bad5a2667a198aabf79acec6a172119ed337f5d163938a5e6f48745',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_dependencies(repo):
    """Existing exact caller manifest is a barrier, never refreshed here."""
    rows = (repo / DEPENDENCIES).read_text().splitlines()
    require(len(rows) == 19, 'exact nineteen caller dependencies required')
    seen = set()
    for row in rows:
        digest, path = row.split('  ', 1)
        require(path not in seen and len(digest) == 64 and
                hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest,
                'stale or duplicate caller dependency: ' + path)
        seen.add(path)


def plan(roster, overlay):
    validate(ROOT, roster)
    require(overlay == {'schema': 'cpu4-trained-g2-test-overlay-v1', 'mode': 'normal',
                       'per_tu_defines': TEST_FLAGS, 'additional_translation_units': OBSERVERS},
            'normal minimal observer overlay mismatch')
    units = copy.deepcopy(roster['translation_units'])
    require(sum(unit['source'] == 'native/o2_optimizer.c' for unit in units) == 1,
            'exact one O2 owner required')
    for unit in units:
        unit['defines'] += TEST_FLAGS.get(unit['source'], [])
    units += copy.deepcopy(OBSERVERS)
    require(len(units) == 37, 'exact37 native TUs required')
    require(sum(unit['source'] == 'src/eshkol_transformer/m3_call_f32_integration.c' for unit in units) == 1,
            'exact one F32 owner required')
    require(len({unit['source'] for unit in units}) == len(units), 'duplicate native owner')
    return units


def preflight(repo, roster, overlay):
    strict_dependencies(repo)
    for path, digest in REQUIRED_REPAIR.items():
        require(hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest,
                'root accepted O2 repair composition/pin refresh required: ' + path)
    units = plan(roster, overlay)
    require(manifest(repo, roster) == (repo / 'native/cpu4_v1_source_members.sha256').read_text(),
            'common source pins stale; root-owned composition/pin refresh required')
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
    # Actual discovered observations, never a replacement freeze or allowlist.
    inventory, paths = {}, set()
    for depfile in sorted(out.rglob('*.d')):
        discovered = dependencies(depfile)
        inventory[depfile.relative_to(out).as_posix()] = [str(path) for path in discovered]
        paths.update(discovered)
    (out / 'observed-dependencies.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (out / 'observed-dependencies.sha256').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path) + '\n' for path in sorted(paths)))


def write_resources(out):
    resources = out / 'resources'
    resources.mkdir()
    for label, tokens in (('train', (3,197,3,41,7)), ('heldout', (197,197,3,41,7))):
        directory = resources / label
        directory.mkdir()
        write_d1_resource(directory, (tokens,), fingerprint=FINGERPRINT, vocab=256)
    hashes(resources, out / 'resource-bytes.sha256')
    return resources / 'train', resources / 'heldout'


def gate(out, runner, cc, cxx, compiler_source):
    require(out.is_absolute() and not out.exists() and not out.is_symlink(), 'fresh absolute output required')
    out.mkdir(mode=0o700)
    program_status, custody_status, before = 1, 0, None

    def run(label, argv, env=None):
        argv = [str(arg) for arg in argv]
        (out / (label + '.argv.json')).write_text(json.dumps(argv) + '\n')
        with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
            try:
                status = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
            except OSError as error:
                stderr.write((repr(error) + '\n').encode())
                status = 127
        (out / (label + '.exit')).write_text(str(status) + '\n')
        if status:
            raise subprocess.CalledProcessError(status, argv)
        return (out / (label + '.stdout')).read_bytes()

    try:
        before = run('source.before', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
        require(not run('source.before.status', ['git', 'status', '--porcelain', '--untracked-files=all']), 'dirty source')
        roster, overlay = load_roster(), json.loads(OVERLAY.read_text())
        units = preflight(ROOT, roster, overlay)
        (out / 'production-inputs.json').write_text(json.dumps(roster, indent=2) + '\n')
        (out / 'test-overlay.json').write_text(json.dumps(overlay, indent=2) + '\n')
        (out / 'selected-native-inputs.json').write_text(json.dumps(units, indent=2) + '\n')
        (out / 'caller-dependencies.sha256').write_bytes((ROOT / DEPENDENCIES).read_bytes())
        (out / 'common-source-members.sha256').write_bytes((ROOT / 'native/cpu4_v1_source_members.sha256').read_bytes())
        (out / 'profile.txt').write_text('candidate=ON\ncompile_jobs=1\nmode=normal\nclosed_modes=two-token,first-eos,second-eos\nrepeat=2-each\nactual_processes=6\n'
            'SAN=NOT RUN\nfresh-process-replay=NOT RUN\ninstalled=NOT RUN\n'
            'closure=actual discovery; independent admission required\n')
        train, heldout = write_resources(out)
        objects, cache = out / 'objects', out / 'cache'
        objects.mkdir()
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
            require(target.stat().st_size > 0 and (objects / (stem + '.d')).stat().st_size > 0,
                    'native object or depfile missing')
            object_paths.append(target)
        library = out / 'libcpu4_resume.a'
        run('archive', ['ar', 'rcsD', library, *object_paths])
        run('archive.members', ['ar', 't', library])
        run('archive.symbols', ['nm', '-a', '--format=posix', library])
        run('archive.imports', ['nm', '-u', '--format=posix', library])
        (out / 'cxx-link').write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(str(cxx)) + ' "$@"\n')
        (out / 'cxx-link').chmod(0o500)
        environment = dict(os.environ)
        for name in ('ESHKOL_PATH', 'ESHKOL_JIT_CACHE_DIR'):
            environment.pop(name, None)
        environment.update(ESHKOL_JIT_CACHE='0', XDG_CACHE_HOME=str(cache), PYTHONDONTWRITEBYTECODE='1',
                           ESHKOL_CXX_COMPILER=str(out / 'cxx-link'), ESHKOL_LIB_DIR=str(ROOT / 'lib'))
        includes = [arg for directory in roster['source_include_dirs'] for arg in ('-I', str(ROOT / directory))]
        raw = out / 'runtime.raw.o'
        run('compile', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '1200s', runner,
            '--strict-types', '--no-stdlib', '--optimize', '0', *includes, '-L', out, '--lib', 'cpu4_resume',
            '--emit-object', '--dump-ir', '--emit-depfile', out / 'runtime.d', CALLER, '-o', raw], environment)
        for suffix in ('', '.ll', '.bc'):
            require(Path(str(raw) + suffix).stat().st_size > 0, 'compiler artifact missing')
        require((out / 'runtime.d').stat().st_size > 0, 'runner depfile missing')
        run('object.imports', ['nm', '-u', '--format=posix', raw])
        run('object.symbols', ['nm', '-a', '--format=posix', raw])
        executable = out / 'runtime'
        run('link', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '120s', runner,
            '--no-stdlib', '-L', out, '--lib', 'cpu4_resume', raw, '-o', executable], environment)
        run('elf.header', ['readelf', '-h', executable])
        run('elf.notes', ['readelf', '-n', executable])
        run('elf.symbols', ['nm', '-a', '--format=posix', executable])
        run('elf.imports', ['nm', '-u', '--format=posix', executable])
        b250_digest = None
        for mode in MODES:
            for repeat in (1, 2):
                environment['ESHKOL_ARENA_POISON'] = '1'
                label = f'{mode}.run{repeat}'
                checkpoint = out / f'{mode}.saved{repeat}.c2'
                final_b250 = out / f'{mode}.final-b250.{repeat}.c2'
                run(label, ['timeout', '--foreground', '--signal=TERM', '--kill-after=5s', '900s',
                            executable, train, heldout, checkpoint, final_b250, mode], environment)
                require(not (out / (label + '.stderr')).read_bytes(), 'runtime diagnostics')
                require(checkpoint.stat().st_size == 25504, 'B125 checkpoint size changed')
                require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == EXPECTED_B125_SHA256,
                        'B125 checkpoint identity changed')
                require(final_b250.stat().st_size == 25504, 'B250 checkpoint size changed')
                observed_b250 = hashlib.sha256(final_b250.read_bytes()).hexdigest()
                require(observed_b250 != EXPECTED_B125_SHA256, 'B250 must differ from B125')
                if b250_digest is None:
                    b250_digest = observed_b250
                    (out / 'final-b250.sha256').write_text(
                        b250_digest + '  ' + final_b250.name + '\n')
                require(observed_b250 == b250_digest, 'B250 checkpoint identity changed')
                check_label = f'{mode}.check{repeat}'
                run(check_label, [sys.executable, ROOT / 'tests/cpu4_trained_g2/check_output.py',
                                 out / (label + '.stdout'), mode], environment)
                require(not (out / (check_label + '.stderr')).read_bytes(), 'checker diagnostics')
            run(mode + '.repeat.stdout', ['cmp', out / f'{mode}.run1.stdout', out / f'{mode}.run2.stdout'])
            run(mode + '.repeat.checkpoint', ['cmp', out / f'{mode}.saved1.c2', out / f'{mode}.saved2.c2'])
            run(mode + '.repeat.final-b250', ['cmp', out / f'{mode}.final-b250.1.c2',
                                               out / f'{mode}.final-b250.2.c2'])
        run('modes.checkpoint', ['cmp', out / 'two-token.saved1.c2', out / 'first-eos.saved1.c2'])
        run('second-eos.checkpoint', ['cmp', out / 'two-token.saved1.c2', out / 'second-eos.saved1.c2'])
        run('modes.final-b250', ['cmp', out / 'two-token.final-b250.1.c2',
                                  out / 'first-eos.final-b250.1.c2'])
        run('second-eos.final-b250', ['cmp', out / 'two-token.final-b250.1.c2',
                                      out / 'second-eos.final-b250.1.c2'])
        program_status = 0
    except subprocess.CalledProcessError as error:
        program_status = error.returncode
    except Exception as error:
        (out / 'driver-error.txt').write_text(repr(error) + '\n')
    finally:
        try:
            inventory_depfiles(out)
            after = run('source.after', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
            require(before is not None and before == after, 'source identity changed')
            require(not run('source.after.status', ['git', 'status', '--porcelain', '--untracked-files=all']), 'dirty source after exit')
        except Exception as error:
            custody_status = 74
            (out / 'custody-error.txt').write_text(repr(error) + '\n')
        (out / 'driver-program.exit').write_text(str(program_status) + '\n')
        (out / 'driver-custody.exit').write_text(str(custody_status) + '\n')
        hashes(out, out / 'SHA256SUMS')
        digest = hashlib.sha256((out / 'SHA256SUMS').read_bytes()).hexdigest()
        (out / 'SHA256SUMS.sha256').write_text(digest + '  SHA256SUMS\n')
    return custody_status or program_status


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit('normal_driver OUTPUT RUNNER CC CXX COMPILER_SOURCE')
    raise SystemExit(gate(*map(Path, sys.argv[1:])))
