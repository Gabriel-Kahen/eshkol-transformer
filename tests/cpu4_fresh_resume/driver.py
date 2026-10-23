"""Distinct training-only external resume gate. Root schedules heavy execution."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
from tests.cpu4_resume import normal_driver as base
from tests.cpu4_fresh_resume import check

ROOT, OVERLAY = base.ROOT, base.OVERLAY
CALLER = ROOT / 'tests/cpu4_fresh_resume/runtime.esk'
DEPENDENCIES = 'tests/cpu4_fresh_resume/DEPENDENCIES.sha256'
load_roster, write_resources = base.load_roster, base.write_resources
require, hashes, inventory_depfiles = base.require, base.hashes, base.inventory_depfiles


def preflight(repo, roster, overlay):
    units = base.preflight(repo, roster, overlay)
    rows = (repo / DEPENDENCIES).read_text().splitlines()
    require(len(rows) == 14, 'exact 14 immutable b0 dependencies required')
    seen = set()
    for row in rows:
        digest, name = row.split('  ', 1)
        require(name not in seen and hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest,
                'stale/duplicate fresh caller dependency: '+name)
        seen.add(name)
    require(len(units) == 36, 'exact inherited 36 TU roster required')
    return units


def external_identity(path):
    record = check.checkpoint(path, 125)
    require(record['sha256'] == check.A125_SHA, 'accepted external A125 SHA required')
    return record['sha256']


class Runner:
    """Direct child PID and bounded process-group custody, including timeout."""
    def __init__(self, out):
        self.out, self.receipts = out, {}

    def __call__(self, label, argv, env=None, seconds=120):
        argv = [str(arg) for arg in argv]
        require(label not in self.receipts, 'duplicate process label')
        (self.out/(label+'.argv.json')).write_text(json.dumps(argv)+'\n')
        record = {'argv': argv, 'pid': None, 'timed_out': False, 'seconds': seconds}
        with (self.out/(label+'.stdout')).open('xb') as stdout, (self.out/(label+'.stderr')).open('xb') as stderr:
            try:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                record['pid'] = process.pid
                try:
                    observed = process.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    record['timed_out'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        observed = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        observed = process.wait()
                record['observed_returncode'] = observed
                status = 124 if record['timed_out'] else observed
            except OSError as error:
                stderr.write((repr(error)+'\n').encode())
                record['os_error'] = repr(error)
                status = 127
        record['status'] = status
        self.receipts[label] = record
        (self.out/(label+'.process.json')).write_text(json.dumps(record, indent=2)+'\n')
        (self.out/(label+'.exit')).write_text(str(status)+'\n')
        if status:
            raise subprocess.CalledProcessError(status, argv)
        return (self.out/(label+'.stdout')).read_bytes()


def build(out, runner, cc, cxx, compiler_source, roster, units, run):
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
            '-c', ROOT / unit['source'], '-o', target], seconds=130)
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
        '--emit-object', '--dump-ir', '--emit-depfile', out / 'runtime.d', CALLER, '-o', raw], environment, seconds=1215)
    for suffix in ('', '.ll', '.bc'):
        require(Path(str(raw) + suffix).stat().st_size > 0, 'compiler artifact missing')
    require((out / 'runtime.d').stat().st_size > 0, 'runner depfile missing')
    run('object.imports', ['nm', '-u', '--format=posix', raw])
    run('object.symbols', ['nm', '-a', '--format=posix', raw])
    executable = out / 'runtime'
    run('link', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '120s', runner,
        '--no-stdlib', '-L', out, '--lib', 'cpu4_resume', raw, '-o', executable], environment, seconds=135)
    run('elf.header', ['readelf', '-h', executable])
    run('elf.notes', ['readelf', '-n', executable])
    run('elf.symbols', ['nm', '-a', '--format=posix', executable])
    run('elf.imports', ['nm', '-u', '--format=posix', executable])
    return executable, environment


def scenarios(out, executable, train, heldout, external, environment, run):
    environment = dict(environment, ESHKOL_ARENA_POISON='1')
    def invoke(label, mode, input_file, prefix, seconds):
        stdout = run(label, [executable, mode, train, heldout, input_file, out/prefix],
                     environment, seconds=seconds)
        require(not (out/(label+'.stderr')).read_bytes(), 'runtime diagnostics: '+label)
        check.transcript(mode, stdout.decode('ascii'))
    invoke('produce', 'produce', external, 'producer', 900)
    require(check.checkpoint(out/'producer.a125.c2', 125)['sha256'] == check.A125_SHA,
            'producer must authenticate external A125 before oracle adoption')
    check.checkpoint(out/'producer.a250.c2', 250)
    invoke('consume', 'consume', external, 'consumer', 900)
    pids = [run.receipts[label]['pid'] for label in ('produce', 'consume')]
    require(all(isinstance(pid, int) and pid > 0 for pid in pids) and len(set(pids)) == 2,
            'distinct observed producer/consumer Popen PIDs required')
    artifacts = check.compare(out, external)
    original = external.read_bytes()
    corrupt = bytearray(original)
    corrupt[-1] ^= 1
    for label, payload in [('corrupt', corrupt), ('truncated', original[:-1])]:
        path = out/(label+'.c2')
        with path.open('xb') as stream:
            stream.write(payload)
        invoke(label, 'reject-corrupt', path, label+'.unused', 30)
    invoke('identity', 'reject-identity', external, 'identity.unused', 30)
    require(not list(out.glob('*.unused*.c2')), 'negative path unexpectedly saved state')
    (out/'oracle.json').write_text(json.dumps({
        'schema': 'cpu4-fresh-resume-v1', 'external_a125_sha256': check.A125_SHA,
        'producer_pid': pids[0], 'consumer_pid': pids[1], 'artifacts': artifacts,
        'actual_training_steps': {'produce': 250, 'consume': 125, 'negative_each': 0},
        'metric_words': list(check.METRICS), 'scope': 'training-only fixed CPU4 N1T4V256'}, indent=2)+'\n')


def gate(out, runner, cc, cxx, compiler_source, external):
    require(out.is_absolute() and not out.exists() and not out.is_symlink(), 'fresh absolute output required')
    external = external.absolute()
    require(out not in external.parents, 'external input cannot be inside fresh output')
    original = external_identity(external)
    out.mkdir(mode=0o700)
    program_status, custody_status, before = 1, 0, None
    run = Runner(out)
    try:
        (out/'external.before.sha256').write_text(original+'  '+str(external)+'\n')
        before = run('source.before', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
        require(not run('source.before.status', ['git', 'status', '--porcelain', '--untracked-files=all']), 'dirty source')
        roster, overlay = load_roster(), json.loads(OVERLAY.read_text())
        units = preflight(ROOT, roster, overlay)
        for name, value in [('production-inputs', roster), ('test-overlay', overlay), ('selected-native-inputs', units)]:
            (out/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
        (out/'caller-dependencies.sha256').write_bytes((ROOT/DEPENDENCIES).read_bytes())
        (out/'common-source-members.sha256').write_bytes((ROOT/'native/cpu4_v1_source_members.sha256').read_bytes())
        (out/'profile.txt').write_text('candidate=ON\ncompile_jobs=1\nproducer_steps=250\nconsumer_steps=125\n'
            'SAN=NOT RUN\ninstalled=NOT RUN\ngeneration=NOT RUN\nCPU-v1-admission=NOT RUN\n')
        train, heldout = write_resources(out)
        executable, environment = build(out, runner, cc, cxx, compiler_source, roster, units, run)
        scenarios(out, executable, train, heldout, external, environment, run)
        program_status = 0
    except subprocess.CalledProcessError as error:
        program_status = error.returncode
    except Exception as error:
        (out/'driver-error.txt').write_text(repr(error)+'\n')
    finally:
        try:
            require(check.digest(external) == original, 'external checkpoint changed')
            require(external_identity(external) == original, 'external checkpoint identity changed')
            (out/'external.after.sha256').write_text(original+'  '+str(external)+'\n')
            inventory_depfiles(out)
            after = run('source.after', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
            require(before is not None and before == after, 'source identity changed')
            require(not run('source.after.status', ['git', 'status', '--porcelain', '--untracked-files=all']), 'dirty source after exit')
            preflight(ROOT, load_roster(), json.loads(OVERLAY.read_text()))
        except Exception as error:
            custody_status = 74
            (out/'custody-error.txt').write_text(repr(error)+'\n')
        (out/'driver-program.exit').write_text(str(program_status)+'\n')
        (out/'driver-custody.exit').write_text(str(custody_status)+'\n')
        hashes(out, out/'SHA256SUMS')
        (out/'SHA256SUMS.sha256').write_text(check.digest(out/'SHA256SUMS')+'  SHA256SUMS\n')
    return custody_status or program_status


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit('driver OUTPUT RUNNER CC CXX COMPILER_SOURCE EXTERNAL_A125_C2')
    raise SystemExit(gate(*map(Path, sys.argv[1:])))
