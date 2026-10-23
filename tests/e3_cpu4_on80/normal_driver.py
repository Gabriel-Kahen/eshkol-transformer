"""Direct uninstalled ON80 E3 fault-matrix: two gradient states, once each."""
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
from tests.e3_cpu4_consumer.resources import generate
from tests.e3_cpu4_on80.source_contract import validate_variant

CALLER = ROOT / 'tests/e3_cpu4_on80/runtime.esk'
DEPENDENCIES = 'tests/e3_cpu4_on80/DEPENDENCIES.sha256'
HELPER = 'tests/e3_cpu4_consumer/helper.c'
MODES = ('absent', 'present')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(repo, roster):
    validate(repo, roster)
    validate_variant(repo)
    require(manifest(repo, roster) == (repo / 'native/cpu4_v1_source_members.sha256').read_text(),
            'common source pins stale; no implicit refresh')
    rows = (repo / DEPENDENCIES).read_text().splitlines()
    seen = set()
    for row in rows:
        digest, path = row.split('  ', 1)
        require(path not in seen and len(digest) == 64 and
                hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest,
                'stale or duplicate dependency: ' + path)
        seen.add(path)
    require(seen == set(DEPENDENCY_PATHS), 'exact caller dependencies differ')
    units = copy.deepcopy(roster['translation_units'])
    require(len(units) == 35 and HELPER not in {u['source'] for u in units},
            'exact original35 production units required')
    units.append({'source': HELPER, 'defines': []})
    return units


DEPENDENCY_PATHS = (
    'tests/e3_cpu4_on80/runtime.esk', 'tests/e3_cpu4_on80/source_contract.py',
    'tests/e3_cpu4_on80/normal_driver.py', 'tests/e3_cpu4_on80/check_output.py',
    'tests/e3_cpu4_consumer/runtime.esk', 'tests/e3_cpu4_consumer/helper.c',
    'tests/e3_cpu4_consumer/resources.py', 'tests/e3_cpu4_consumer/observation.py',
    'tests/e3_cpu4_consumer/test_source.py', 'tests/e3_cpu4_consumer/host_reader.py',
    'tests/d2/test_resources.py', 'tests/t1/reference.py',
    'tests/cpu4_v1_aggregate/closure.py',
)


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
    generate(resources)
    hashes(resources, out / 'resource-bytes.sha256')
    return resources / 'corpus'


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
        roster = load_roster()
        units = preflight(ROOT, roster)
        (out / 'production-inputs.json').write_text(json.dumps(roster, indent=2) + '\n')
        (out / 'selected-native-inputs.json').write_text(json.dumps(units, indent=2) + '\n')
        (out / 'caller-dependencies.sha256').write_bytes((ROOT / DEPENDENCIES).read_bytes())
        (out / 'common-source-members.sha256').write_bytes((ROOT / 'native/cpu4_v1_source_members.sha256').read_bytes())
        (out / 'profile.txt').write_text('candidate=ON\nparameter_control_bytes=80\ncompile_jobs=1\nmode=normal\n'
            'closed_modes=absent,present\nrepeat=NOT RUN\nactual_processes=2\n'
            'SAN=NOT RUN\ncomplete_retention_census=BLOCKED\ninstalled=NOT RUN\n'
            'closure=actual discovery; independent admission required\n')
        corpus = write_resources(out)
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
            require(not (out / ('native.' + stem + '.stderr')).read_bytes(), 'native compiler diagnostics')
            object_paths.append(target)
        library = out / 'libe3_cpu4_on80.a'
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
            '--strict-types', '--no-stdlib', '--optimize', '0', *includes, '-L', out, '--lib', 'e3_cpu4_on80',
            '--emit-object', '--dump-ir', '--emit-depfile', out / 'runtime.d', CALLER, '-o', raw], environment)
        for suffix in ('', '.ll', '.bc'):
            require(Path(str(raw) + suffix).stat().st_size > 0, 'compiler artifact missing')
        require((out / 'runtime.d').stat().st_size > 0, 'runner depfile missing')
        run('object.imports', ['nm', '-u', '--format=posix', raw])
        run('object.symbols', ['nm', '-a', '--format=posix', raw])
        executable = out / 'runtime'
        run('link', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '120s', runner,
            '--no-stdlib', '-L', out, '--lib', 'e3_cpu4_on80', raw, '-o', executable], environment)
        run('elf.header', ['readelf', '-h', executable])
        run('elf.notes', ['readelf', '-n', executable])
        run('elf.symbols', ['nm', '-a', '--format=posix', executable])
        run('elf.imports', ['nm', '-u', '--format=posix', executable])
        for mode in MODES:
            environment['ESHKOL_ARENA_POISON'] = '1'
            label = mode + '.run1'
            run(label, ['timeout', '--foreground', '--signal=TERM', '--kill-after=5s', '120s',
                        executable, corpus, mode], environment)
            require(not (out / (label + '.stderr')).read_bytes(), 'runtime diagnostics')
            run(mode + '.check1', [sys.executable, '-B', ROOT / 'tests/e3_cpu4_on80/check_output.py',
                                  out / (label + '.stdout'), mode], environment)
            require(not (out / (mode + '.check1.stderr')).read_bytes(), 'checker diagnostics')
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
