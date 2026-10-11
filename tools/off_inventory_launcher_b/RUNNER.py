#!/usr/bin/python3
"""One-use, fail-closed F/G OFF inventory comparison. Never executes an object.

An external ROOT-reviewed launch supplies the SHA-256 of a separate binding
file. Its exact digest is a trust input, not a value learned from that file.
"""
import posix  # BuiltinImporter resolves this before the zip search entry.
import sys


def preflight_absent_stdlib_zip(path='/usr/lib/python314.zip'):
    """Fail before runner imports if the first Python search entry is present."""
    parts = path.split('/')
    if not (parts[0] == '' and len(parts) >= 3 and
            all(part not in ('', '.', '..') for part in parts[1:])):
        raise RuntimeError('stdlib archive path is not canonical absolute')
    if posix.geteuid() == 0:
        raise RuntimeError('stdlib archive trust needs non-root caller')
    try:
        posix.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError('stdlib archive search entry present')
    parent = ''
    for part in parts[1:-1]:
        parent += '/' + part
        st = posix.lstat(parent)
        if st.st_uid != 0 or st.st_mode & 0o022 or st.st_mode & 0o170000 != 0o040000:
            raise RuntimeError(f'stdlib archive parent mutable: {parent}')
    root = posix.lstat('/')
    if root.st_uid != 0 or root.st_mode & 0o022 or root.st_mode & 0o170000 != 0o040000:
        raise RuntimeError('stdlib archive root mutable')


if __name__ == '__main__':
    preflight_absent_stdlib_zip()

import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import stat
import subprocess
import time

BASE = Path('/home/gabe/.codex/evidence/eshkol-transformer')
STEM = 'g3r-cold-s-dual64-tagged-native-off-'
PLAN = BASE / (STEM + 'parity-plan-20261011-b')
TOOL = BASE / (STEM + 'parity-tool-closure-20261011-d')
PARSER = BASE / (STEM + 'inventory-parser-20261011-d')
RUN = {s: BASE / (STEM + f'build-run-20261011-{s}') for s in ('fb', 'gb')}
ROOT_SHA = {'fb': 'ffbe5393a90f12134fb64693511e9d99ed50e38940032e61fec7d777c58a81d3',
            'gb': 'a02d8a0897904311641e4fc7f7f3eb32c6bc6d1fce39c7edf852cb2df92d7e34'}
PINS = {
    PLAN / 'SHA256SUMS-PLAN': '5eed63c2b427d52ad07c4e1e3a3f86f38c1ba077b097860f1892f0f68da96afd',
    BASE / (STEM + 'parity-independent-plan-review-20261011-b/SHA256SUMS-REVIEW'):
        'd8d9406816e24814ba0dc0d8fa2ad148a19be0f8929516b616a0e202ea133b1c',
    BASE / (STEM + 'parity-root-plan-review-20261011-b/SHA256SUMS-ROOT-REVIEW'):
        '5116e363dba86ca29eb87ec763985f5f0803a2ae80c90e910ad5c492606a0b7c',
    TOOL / 'SHA256SUMS-CLOSURE': 'b128967e98ba884eba81d337b198dbc731c90a623847fb2b85598c1f2a37d344',
    BASE / (STEM + 'parity-tool-closure-review-20261011-d/SHA256SUMS-REVIEW'):
        'ac67b821156a1ec24e8d275f6dc6fad2cde966f7b1b667d289e3ea97d0346f0a',
    PARSER / 'SHA256SUMS-PARSER': 'c89a991654ea852b67301b6feeacb9702bd4d570f38a3a35220ecc4e7804e970',
    BASE / (STEM + 'inventory-parser-independent-review-20261011-d/SHA256SUMS-REVIEW'):
        '880f2c369973ae586757cdb45341e99357d24f830db6228eb2c51126b36039b6',
    BASE / (STEM + 'parity-tool-closure-root-review-20261011-d/SHA256SUMS-ROOT-REVIEW'):
        '66cb647a1b238c840ac4080e4613f859f8b156c40e60956032b0b7e332c67e84',
    BASE / (STEM + 'inventory-parser-root-review-20261011-d/SHA256SUMS-ROOT-REVIEW'):
        '2631dcb9b9c3e91bfecb457b0d3d8b867fb5848fc8ba05d4a3da5236867c77a7',
    BASE / (STEM + 'config-run-20261011-fb/SHA256SUMS-RUN'):
        '9fe938db7abb7858d4de7c7b91953a47858be110d6d8a953aa5c1ef74ab18260',
    BASE / (STEM + 'config-run-20261011-gb/SHA256SUMS-RUN'):
        'f5acfceb9910dea3546149c383610d36fd6762e6464e3c8c6bf72643d41a898b',
    BASE / (STEM + 'graph-audit-20261011-a/SHA256SUMS-AUDIT'):
        '340c92bbd7540e6ddf6c977f5fddbec3d656f10833a4f8cec95c0b41c4500c98',
    BASE / (STEM + 'build-plan-20261011-fb/SHA256SUMS-PLAN'):
        '66b4d8e5ab1410d6bef533da88cfe998a4a97eb847e24af0a67e5052aebda4e2',
    BASE / (STEM + 'build-plan-20261011-gb/SHA256SUMS-PLAN'):
        '2806213768bfdae0a4fa8a09b46ef7bc5a74c2d315496cec7f8c2f5cd6fd5d28',
}
RUNNER_ROOT = Path(__file__).resolve().parent
FORBIDDEN_OFF = (
    'eshkol_dual8_tagged_construct_v1', 'eshkol_dual8_tagged_read_v1',
    'eshkol_dual8_tagged_validate_v1', 'eshkol_dual8_tagged_read_or_abort_v1',
    'eshkol_dual8_tagged_profile_v1', 'eshkol_dual8_tagged_test_start_v1',
)
CPU_SECONDS = 300
WALL_SECONDS = 360
AS_BYTES = 4 * 1024 ** 3
FSIZE_BYTES = 1024 ** 3
RELEASE_ROOT = BASE / (STEM + 'inventory-runner-release-20261011-j')
RELEASE_PATH = RELEASE_ROOT / 'RELEASE.json'
EXCEPTION_SEAL = RELEASE_ROOT / 'SHA256SUMS-EXCEPTIONS'
BINDING_PATH = BASE / (STEM + 'inventory-runner-root-binding-20261011-j') / 'PINS.json'
REVIEW_AUTHORITY = {
    'independent': BASE / (STEM + 'inventory-runner-independent-review-20261011-j') /
                   'SHA256SUMS-REVIEW',
    'root': BASE / (STEM + 'inventory-runner-root-review-20261011-j') /
            'SHA256SUMS-ROOT-REVIEW',
}
_OWN_MANIFEST = None
SAFE_IMPORT_PATH = ('/usr/lib/python314.zip', '/usr/lib/python3.14',
                    '/usr/lib/python3.14/lib-dynload')
STDLIB_ROOT = Path('/usr/lib/python3.14')


class Hold(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def freeze_import_resolution():
    require(SAFE_IMPORT_PATH == ('/usr/lib/python314.zip', '/usr/lib/python3.14',
                                 '/usr/lib/python3.14/lib-dynload') and
            sys.flags.isolated == 1 and sys.flags.no_site == 1 and
            sys.flags.ignore_environment == 1 and tuple(sys.path) == SAFE_IMPORT_PATH,
            'unreviewed Python import path or startup mode')
    require([(getattr(finder, '__module__', None), getattr(finder, '__name__', None))
             for finder in sys.meta_path] ==
            [('_frozen_importlib', 'BuiltinImporter'),
             ('_frozen_importlib', 'FrozenImporter'),
             ('_frozen_importlib_external', 'PathFinder')],
            'unreviewed Python import finder')
    preflight_absent_stdlib_zip()
    trusted_root_owned_path(STDLIB_ROOT)
    trusted_root_owned_path(STDLIB_ROOT / 'lib-dynload')


def trusted_root_owned_path(path):
    """Trust boundary: root alone may change this installed runtime path."""
    path = Path(path)
    require(path.is_absolute() and os.geteuid() != 0,
            'stdlib trust needs absolute path and non-root caller')
    for selected in (path, path.resolve(strict=True)):
        current = Path('/')
        root_stat = current.lstat()
        require(root_stat.st_uid == 0 and not (root_stat.st_mode & 0o022),
                'stdlib root directory writable outside root')
        for part in selected.parts[1:]:
            current /= part
            st = current.lstat()
            require(st.st_uid == 0 and not (st.st_mode & 0o022),
                    f'stdlib path writable outside root: {current}')
        require(stat.S_ISREG(current.stat().st_mode) or
                stat.S_ISDIR(current.stat().st_mode), 'stdlib trust path not file or directory')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    selected = str(Path(path))
    cur, chain, seen = Path(path), [], set()
    for _ in range(32):
        require(str(cur) not in seen, 'symlink cycle')
        seen.add(str(cur))
        mode = cur.lstat().st_mode
        if stat.S_ISLNK(mode):
            target = os.readlink(cur)
            chain.append({'path': str(cur), 'target': target})
            cur = Path(os.path.normpath(os.path.join(cur.parent, target)))
            continue
        require(stat.S_ISREG(mode), 'runtime import not regular')
        return {'selected': selected, 'chain': chain, 'resolved': str(cur),
                'bytes': cur.stat().st_size, 'mode': stat.S_IMODE(mode),
                'sha256': sha_file(cur)}
    raise Hold('symlink chain too long')


def relative(name):
    p = Path(name)
    require(not p.is_absolute() and p.parts and all(x not in ('', '.', '..') for x in p.parts),
            'unsafe relative path')
    return p


def regular(path):
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, f'nonregular or linked input: {path}')
    return st


def within(path, root):
    require(path.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
            'path escapes reviewed root')


def seal_entries(path, raw=None):
    raw = path.read_bytes() if raw is None else raw
    rows = {}
    for line in raw.splitlines():
        m = re.fullmatch(rb'([0-9a-f]{64})  (.+)', line)
        require(m is not None, f'malformed seal: {path}')
        name = m.group(2).decode('utf-8', 'strict')
        relative(name)
        require(name not in rows, f'duplicate seal member: {name}')
        rows[name] = m.group(1).decode()
    require(rows, f'empty seal: {path}')
    return rows


def verify_seal(path, expected_sha):
    regular(path)
    raw = path.read_bytes()
    require(sha_bytes(raw) == expected_sha, f'seal hash drift: {path}')
    rows = seal_entries(path, raw)
    for name, want in rows.items():
        member = path.parent / relative(name)
        within(member, path.parent)
        regular(member)
        require(sha_file(member) == want, f'sealed member drift: {member}')
    return rows


def load_pinned(path, name, expected_sha):
    regular(path)
    raw = path.read_bytes()
    require(sha_bytes(raw) == expected_sha, f'Python source drift: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def read_pinned_json(path, expected_sha, root=None):
    if root is not None:
        within(path, root)
    regular(path)
    raw = path.read_bytes()
    require(len(raw) <= 2 * 1024 * 1024 and sha_bytes(raw) == expected_sha,
            f'sealed JSON drift: {path}')
    return json.loads(raw)


def trusted_binding(expected_sha):
    require(isinstance(expected_sha, str) and re.fullmatch(r'[0-9a-f]{64}', expected_sha),
            'external ROOT-reviewed binding digest absent')
    regular(BINDING_PATH)
    raw = BINDING_PATH.read_bytes()
    require(sha_bytes(raw) == expected_sha, 'external binding digest drift')
    binding = json.loads(raw)
    keys = {'schema', 'runner_manifest_sha256', 'release_sha256',
            'exception_manifest_sha256', 'exception_file_sha256',
            'independent_review_seal_sha256', 'root_review_seal_sha256'}
    require(isinstance(binding, dict) and set(binding) == keys and
            binding['schema'] == 'eshkol-off-inventory-binding-v1',
            'external binding schema drift')
    for key in keys - {'schema'}:
        require(isinstance(binding[key], str) and re.fullmatch(r'[0-9a-f]{64}', binding[key]),
                f'external binding {key} invalid')
    require(binding['independent_review_seal_sha256'] !=
            binding['root_review_seal_sha256'], 'review roles share one seal digest')
    return binding


def review_gate(release_path, binding_sha=None):
    global _OWN_MANIFEST
    freeze_import_resolution()
    binding = trusted_binding(binding_sha)
    require(set(REVIEW_AUTHORITY) == {'independent', 'root'} and
            REVIEW_AUTHORITY['independent'] != REVIEW_AUTHORITY['root'] and
            all(path.is_absolute() for path in REVIEW_AUTHORITY.values()),
            'review authorities must be distinct canonical paths')
    require(release_path == RELEASE_PATH, 'release path is not canonical')
    regular(release_path)
    raw_release = release_path.read_bytes()
    require(sha_bytes(raw_release) == binding['release_sha256'],
            'release file differs from external binding')
    release = json.loads(raw_release)
    require(set(release) == {'schema', 'runner_manifest_sha256', 'exception_file_sha256'},
            'release schema drift')
    require(release['schema'] == 'eshkol-off-inventory-release-v3' and
            release['runner_manifest_sha256'] == binding['runner_manifest_sha256'] and
            release['exception_file_sha256'] == binding['exception_file_sha256'],
            'release schema or exception digest drift')
    runner_manifest = RUNNER_ROOT / 'SHA256SUMS-RUNNER'
    runner_rows = verify_seal(runner_manifest, binding['runner_manifest_sha256'])
    exception_rows = verify_seal(EXCEPTION_SEAL, binding['exception_manifest_sha256'])
    require(exception_rows == {'EXCEPTIONS.json': binding['exception_file_sha256']},
            'exception manifest differs from external binding')
    own = read_pinned_json(RUNNER_ROOT / 'MANIFEST.json', runner_rows['MANIFEST.json'],
                           RUNNER_ROOT)
    require(own['schema'] == 'eshkol-off-inventory-runner-v9', 'runner manifest schema')
    require(identity(sys.executable) == own['interpreter'] and sys.version == own['python_version'],
            'runner interpreter drift')
    for name, row in own['runtime_imports'].items():
        if Path(row['selected']).is_relative_to(STDLIB_ROOT):
            trusted_root_owned_path(row['selected'])
        require(identity(row['selected']) == row, f'runner import drift: {name}')
    for path, row in own['linked_libraries'].items():
        require(identity(path) == row, f'runner linked library drift: {path}')
    trusted_review_decisions(release, binding)
    _OWN_MANIFEST = own
    return release


def trusted_review_decisions(release, binding):
    for role, seal in REVIEW_AUTHORITY.items():
        digest = binding[f'{role}_review_seal_sha256']
        rows = verify_seal(seal, digest)
        require('RELEASE.json' in rows, f'{role} sealed release missing')
        decision = seal.parent / 'RELEASE.json'
        review_raw = decision.read_bytes()
        require(sha_bytes(review_raw) == rows['RELEASE.json'],
                f'{role} review decision drift after seal check')
        review = json.loads(review_raw)
        require(review == {'role': role, 'verdict': 'RELEASE',
                           'runner_manifest_sha256': binding['runner_manifest_sha256'],
                           'release_sha256': binding['release_sha256'],
                           'exception_manifest_sha256': binding['exception_manifest_sha256'],
                           'exception_file_sha256': binding['exception_file_sha256']},
                f'{role} review did not release exact runner and exceptions')


def check_imports():
    own = _OWN_MANIFEST
    require(own is not None, 'runner manifest not authenticated')
    allowed = {row['selected']: row for row in own['runtime_imports'].values()}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path is None:
            continue
        path = str(Path(path))
        if Path(path).is_relative_to(STDLIB_ROOT):
            trusted_root_owned_path(path)
        require(path in allowed and identity(path) == allowed[path],
                f'unreviewed or changed Python import: {name}')


class PinnedRunReads:
    """Serve reviewed RUN metadata bytes to sealed Plan/closure routines."""
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def __enter__(self):
        self.old_text, self.old_bytes = Path.read_text, Path.read_bytes
        snapshot, old_text, old_bytes = self.snapshot, self.old_text, self.old_bytes

        def read_text(path, encoding=None, errors=None):
            raw = snapshot.get(str(path))
            return (raw.decode(encoding or 'utf-8', errors or 'strict') if raw is not None
                    else old_text(path, encoding=encoding, errors=errors))

        def read_bytes(path):
            raw = snapshot.get(str(path))
            return raw if raw is not None else old_bytes(path)

        Path.read_text, Path.read_bytes = read_text, read_bytes
        return self

    def __exit__(self, *_):
        Path.read_text, Path.read_bytes = self.old_text, self.old_bytes


def snapshot_run_metadata(side):
    """Hash and retain one buffer for every RUN member except product bytes."""
    seal = RUN[side] / 'SHA256SUMS-RUN'
    regular(seal)
    seal_raw = seal.read_bytes()
    require(sha_bytes(seal_raw) == ROOT_SHA[side], f'{side}: RUN root drift')
    rows = seal_entries(seal, seal_raw)
    products = {name for name in rows if name.startswith('build/') and
                name.endswith(('.o', '.a'))}
    snapshot, total = {}, 0
    for name, expected in rows.items():
        if name in products:
            continue
        path = RUN[side] / relative(name)
        within(path, RUN[side])
        remaining = 256 * 1024 * 1024 - total
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, 'rb') as stream:
            st = os.fstat(stream.fileno())
            require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and
                    st.st_size <= remaining, f'{side}: RUN metadata snapshot cap: {name}')
            raw = stream.read(remaining + 1)
        total += len(raw)
        require(len(raw) == st.st_size and total <= 256 * 1024 * 1024 and
                sha_bytes(raw) == expected,
                f'{side}: RUN metadata snapshot drift/cap: {name}')
        snapshot[str(path)] = raw
    return products, snapshot


def authenticate_metadata():
    """Metadata admission only: this function does not open product bytes."""
    pinned_rows = {}
    for seal, expected in PINS.items():
        pinned_rows[seal.parent] = verify_seal(seal, expected)
    for side in ('fb', 'gb'):
        regular(RUN[side] / 'SHA256SUMS-RUN')
        require(sha_file(RUN[side] / 'SHA256SUMS-RUN') == ROOT_SHA[side], 'RUN root drift')
    snapshots = {}
    product_names = {}
    for side in ('fb', 'gb'):
        product_names[side], side_snapshot = snapshot_run_metadata(side)
        snapshots.update(side_snapshot)
    # Sealed Plan B routines authenticate exact 155 child routes/mounts and
    # measured source closure without consulting ELF/object/archive contents.
    freeze_import_resolution()
    plan = load_pinned(PLAN / 'COMPARE_PLAN.py', 'COMPARE_PLAN',
                       '05c3e7d5da0a555eefd3a94c1fd6fced78fd1bb9a6eba3df362844531cbcc3d4')
    source = load_pinned(PLAN / 'SOURCE_CLOSURE.py', 'SOURCE_CLOSURE',
                         'f81f93a51d22942eec8f133f1cff63fd0c869342afe84f4c77176148d172e543')
    with PinnedRunReads(snapshots):
        left, right = plan.load_side('fb'), plan.load_side('gb')
        pairs = plan.pair_plan(left, right)
        closure = {'source_delta': source.source_delta()}
        closure['object_closure'], closure['dependency_input_pins'] = source.closure(
            closure['source_delta'])
    require(pairs == read_pinned_json(PLAN / 'PAIR-MAP.json',
                                     pinned_rows[PLAN]['PAIR-MAP.json'], PLAN)['pairs'],
            'pair map drift')
    require(closure == read_pinned_json(PLAN / 'SOURCE-CLOSURE.json',
                                        pinned_rows[PLAN]['SOURCE-CLOSURE.json'], PLAN),
            'source closure drift')
    require(len(pairs) == 155 and sum(p['kind'] == 'object' for p in pairs) == 152
            and sum(p['kind'] == 'archive' for p in pairs) == 3, 'pair census drift')
    for side, products in (('fb', left[0]), ('gb', right[0])):
        expected = {f'build/{p["path"]}' for p in products}
        require(expected == product_names[side], f'{side}: snapshot product census drift')
        seal_raw = (RUN[side] / 'SHA256SUMS-RUN').read_bytes()
        require(sha_bytes(seal_raw) == ROOT_SHA[side], f'{side}: RUN root changed')
        declared = seal_entries(RUN[side] / 'SHA256SUMS-RUN', seal_raw)
        require(expected == {x for x in declared if x.endswith(('.o', '.a'))},
                f'{side}: manifest product path census drift')
        for p in products:
            path = RUN[side] / 'build' / relative(p['path'])
            st = regular(path)  # lstat only; no product bytes opened
            require(st.st_size == p['bytes'] and p['bytes'] <= 160 * 1024 * 1024,
                    f'{side}: product byte count/cap drift: {p["path"]}')
            require(declared[f'build/{p["path"]}'] == p['sha256'],
                    f'{side}: product hash map drift')
        authenticate_run_metadata(side, expected)
    return plan, pairs, closure, {'fb': left, 'gb': right}


def verify_tools():
    freeze_import_resolution()
    importlib.import_module('sysconfig')
    check_imports()  # Authenticate the transitive stdlib import before Tool source executes.
    tool_seal = TOOL / 'SHA256SUMS-CLOSURE'
    tool_rows = verify_seal(tool_seal, PINS[tool_seal])
    tool = load_pinned(TOOL / 'TOOL_CLOSURE.py', 'TOOL_CLOSURE',
                       'd7c6c1ce258f26fd310dcba4ead8251308a834c91bc7d0c1e1ce5495c34907f3')
    record = read_pinned_json(TOOL / 'CLOSURE.json', tool_rows['CLOSURE.json'], TOOL)
    require(record['schema'] == 1 and record['inventory_parser']['status'] == 'MISSING_HOLD',
            'tool closure schema drift')
    require(tool.identity(record['image']['os_release']['selected']) == record['image']['os_release']
            and record['image']['kernel'] == os.uname().release, 'host image pin drift')
    require(sha_file('/proc/self/mountinfo') == record['image']['mountinfo_sha256'],
            'mount namespace pin drift')
    require(record['python_runtime']['version'] == sys.version
            and tool.identity(sys.executable) == record['python_runtime']['executable'],
            'Python runtime pin drift')
    require(tool.identity(record['loader']['selected']) == record['loader'], 'loader pin drift')
    for row in record['tools'].values():
        require(tool.identity(row['selected']) == row, 'tool pin drift')
    for path, row in record['libraries'].items():
        require(tool.identity(path) == row, 'library pin drift')
    for row in record['python_imports'].values():
        if 'sha256' in row:
            require(tool.identity(row['selected']) == row, 'Python import pin drift')
    parser_path = PARSER / 'INVENTORY.py'
    require(sha_file(parser_path) == '4944bc67066bf55c26e9da4a3fa075458b4c5f783164a88e0730d5ed9f6f23a4',
            'parser source pin drift')
    return record


def authenticate_run_files(side):
    """Open/hash complete RUN manifest only after review release and admission."""
    seal = RUN[side] / 'SHA256SUMS-RUN'
    regular(seal)
    raw = seal.read_bytes()
    require(sha_bytes(raw) == ROOT_SHA[side], f'{side}: RUN root changed')
    rows = seal_entries(seal, raw)
    for name, want in rows.items():
        path = RUN[side] / relative(name)
        within(path, RUN[side])
        regular(path)
        require(sha_file(path) == want, f'{side}: RUN member hash drift: {name}')
    return len(rows)


def authenticate_run_metadata(side, product_names):
    """Recheck every sealed RUN member except product bytes before child admission."""
    seal = RUN[side] / 'SHA256SUMS-RUN'
    regular(seal)
    raw = seal.read_bytes()
    require(sha_bytes(raw) == ROOT_SHA[side], f'{side}: RUN root changed')
    rows = seal_entries(seal, raw)
    require(product_names <= set(rows), f'{side}: product manifest gap')
    for name, expected_sha in rows.items():
        if name in product_names:
            continue
        path = RUN[side] / relative(name)
        within(path, RUN[side])
        regular(path)
        require(sha_file(path) == expected_sha, f'{side}: RUN metadata drift: {name}')
    return rows


def read_run_json(side, name):
    require(side in RUN, 'invalid RUN side')
    seal = RUN[side] / 'SHA256SUMS-RUN'
    regular(seal)
    raw = seal.read_bytes()
    require(sha_bytes(raw) == ROOT_SHA[side], f'{side}: RUN root changed')
    rows = seal_entries(seal, raw)
    require(name in rows, f'{side}: unsealed RUN metadata: {name}')
    return read_pinned_json(RUN[side] / relative(name), rows[name], RUN[side])


def archive_order(command, archive, product_paths):
    require(len(command) == 3 and command[:2] == ['/usr/bin/bash', '-c'],
            'archive child shell route drift')
    text = command[2]
    prefix = f' && /usr/bin/ar qc {archive}  '
    suffix = f' && /usr/bin/ranlib {archive} && :'
    require(text.count(prefix) == 1 and text.endswith(suffix), 'archive command shape drift')
    words = shlex.split(text.split(prefix, 1)[1][:-len(suffix)])
    require(words and len(words) == len(set(words)) and all(x in product_paths for x in words),
            'archive command member drift')
    names = [Path(x).name for x in words]
    require(len(names) == len(set(names)), 'duplicate archive member basename')
    return words


def index_members(archive, expected_paths, products):
    members = archive['members']
    require([m['name'] for m in members] == [Path(p).name for p in expected_paths],
            'archive member order/name drift')
    offsets = {m['header_offset']: m['name'] for m in members}
    require(len(offsets) == len(members), 'archive member offset collision')
    for member, object_path in zip(members, expected_paths):
        require(member['raw_sha256'] == products[object_path]['sha256']
                and member['size'] == products[object_path]['bytes'],
                'archive member raw bytes differ from standalone object')
    expected_index = []
    for m in members:
        for sym in m['elf']['defined_symbols']:
            if sym['name'] and sym['binding'] in (1, 2, 10):
                expected_index.append((sym['name'], m['name']))
    actual_index = [(r['symbol'], r['member']) for r in archive['symbol_index']]
    require(actual_index == expected_index, 'archive index order/completeness drift')
    require(len(archive['headers']) == len(members) +
            (2 if any(h['kind'] == 'long_names' for h in archive['headers']) else 1),
            'archive pseudo-member census drift')
    require(archive['headers'][0]['kind'] == 'index', 'archive index header position drift')
    defined = [s for m in members for s in m['elf']['defined_symbols']]
    undefined = [s for m in members for s in m['elf']['undefined_symbols']]
    require(not any(s['name'] in FORBIDDEN_OFF for s in defined), 'OFF archive defines tagged-only symbol')
    return {'defined_count': len(defined), 'undefined_count': len(undefined),
            'defined_sha256': sha_bytes(json.dumps(sorted(defined, key=repr),
                                                    sort_keys=True).encode()),
            'undefined_sha256': sha_bytes(json.dumps(sorted(undefined, key=repr),
                                                      sort_keys=True).encode()),
            'index': actual_index, 'member_names': [m['name'] for m in members]}


def normalize_symbol_name(name, roots):
    for old, new in roots:
        if old in name:
            require(name.startswith(old) and (len(name) == len(old) or name[len(old)] == '/'),
                    'embedded or adjacent path in symbol')
            value = new + name[len(old):]
            require(not any(path in value for path, _ in roots),
                    'multiple measured paths in symbol')
            return value
    return name


def debug_bytes(raw, section, roots):
    """Normalize only whole NUL-terminated paths in recognized debug string tables."""
    if section['name'] not in ('.debug_str', '.debug_line_str'):
        return section['raw_sha256']
    blob = raw[section['offset']:section['offset'] + section['size']]
    if not blob and section['size'] == 0:
        return sha_bytes(b'')
    require(len(blob) == section['size'] and blob.endswith(b'\0'), 'malformed debug string table')
    out = []
    for item in blob[:-1].split(b'\0'):
        value = item.decode('latin-1')
        out.append(normalize_symbol_name(value, roots).encode('latin-1'))
    return sha_bytes(b'\0'.join(out) + b'\0')


def canonical_elf(inv, raw, roots):
    """Conservative index renumbering; any ambiguous section identity holds."""
    rows = inv['sections']
    groups = {g['section']: g['signature'] for g in inv['groups']}
    member_groups = {}
    for g in inv['groups']:
        for index in g['members']:
            require(index not in member_groups, 'section belongs to multiple groups')
            member_groups[index] = g['signature']
    keys = {}
    for s in rows:
        key = (s['name'], s['type'], groups.get(s['index']), member_groups.get(s['index']))
        require(key not in keys.values(), 'ambiguous duplicate section identity')
        keys[s['index']] = key
    def key(index):
        require(index in keys, 'invalid section reference')
        return keys[index]
    sections = []
    for s in rows:
        out = {k: v for k, v in s.items() if k not in ('index', 'link', 'info', 'raw_hex', 'raw_sha256')}
        out['identity'] = key(s['index'])
        out['link'] = key(s['link']) if s['link'] else None
        out['info'] = (groups[s['index']] if s['type'] == 17 else
                       key(s['info']) if s['type'] in (4, 9) or s['flags'] & 0x40
                       else s['info'])
        if s['type'] in (2, 4, 9, 11, 17):
            out['raw_sha256'] = '<STRUCTURED>'
        else:
            out['raw_sha256'] = (debug_bytes(raw, s, roots) if s['name'].startswith('.debug')
                                 else s['raw_sha256'])
        sections.append(out)
    symbols = []
    for s in inv['defined_symbols'] + inv['undefined_symbols']:
        out = dict(s)
        out['name'] = normalize_symbol_name(s['name'], roots) if s['type'] == 4 else s['name']
        out['section'] = key(s['section']) if s['section'] < len(rows) else s['section']
        out['table'] = key(s['table'])
        if out['version'] is not None:
            out['version'] = {k: v for k, v in out['version'].items() if k != 'index'}
        # Symbol table indices are normalized through names only when unique.
        out.pop('index')
        symbols.append(out)
    symbol_keys = [(s['table'], s['name'], s['binding'], s['type'], s['section']) for s in symbols]
    require(len(symbol_keys) == len(set(map(repr, symbol_keys))), 'ambiguous duplicate symbol identity')
    relocs = []
    for r in inv['relocations']:
        out = {k: v for k, v in r.items() if k not in ('section', 'target_section', 'symbol_table', 'symbol_index')}
        out.update(section=key(r['section']), target_section=key(r['target_section']),
                   symbol_table=key(r['symbol_table']))
        relocs.append(out)
    group_rows = [{'flag': g['flag'], 'signature': g['signature'],
                   'members': [key(x) for x in g['members']]} for g in inv['groups']]
    canonical = {'header': {k: inv[k] for k in ('ident_hex', 'osabi', 'abi_version', 'class',
                                          'endian', 'machine', 'type', 'flags')},
            'sections': sorted(sections, key=lambda x: repr(x['identity'])),
            'groups': sorted(group_rows, key=lambda x: x['signature']),
            'symbols': sorted(symbols, key=repr),
            'relocations': sorted(relocs, key=repr),
            'version_records': sorted(({**r, 'section': key(r['section'])}
                                       for r in inv['version_records']), key=repr)}
    return json.loads(json.dumps(canonical, sort_keys=True))


def delta_rows(path, f, g, closure):
    rows = []
    def add(field, identity, left, right):
        if left != right:
            rows.append({'object': path, 'field': field, 'identity': identity,
                         'f': left, 'g': right,
                         'source_changed': closure['source_changed']})
    add('header', None, f['header'], g['header'])
    fields = {
        'sections': lambda r: repr(r['identity']),
        'groups': lambda r: repr(r['signature']),
        'symbols': lambda r: repr((r['table'], r['name'], r['type'], r['section'])),
        'relocations': lambda r: repr((r['target_section'], r['offset'], r['type'], r['symbol'])),
        'version_records': lambda r: repr((r['section'], r['kind'])),
    }
    for field, identity in fields.items():
        left = {identity(r): r for r in f[field]}
        right = {identity(r): r for r in g[field]}
        require(len(left) == len(f[field]) and len(right) == len(g[field]),
                f'ambiguous {field} identity')
        for ident in sorted(set(left) | set(right)):
            add(field, ident, left.get(ident), right.get(ident))
    return rows


def changed_lines(diff, expected):
    """Exact added-line locations from the authenticated F/G Git trees."""
    result, path, seen = {name: set() for name in expected}, None, set()
    for line in diff.decode('utf-8', 'strict').splitlines():
        if line.startswith('diff --git '):
            match = re.fullmatch(r'diff --git a/(\S+) b/\1', line)
            require(match is not None and match.group(1) in result,
                    'unrecognized changed source path')
            path = match.group(1)
            require(path not in seen, 'duplicate changed source path')
            seen.add(path)
        elif line.startswith('@@ '):
            require(path is not None, 'source hunk without path')
            match = re.match(r'@@ -[0-9]+(?:,[0-9]+)? \+([0-9]+)(?:,([0-9]+))? @@', line)
            require(match is not None, 'malformed source hunk')
            first, count = int(match.group(1)), int(match.group(2) or '1')
            require(count <= 100000 and first >= 0, 'source hunk bounds')
            result[path].update(range(first, first + count))
    require(seen == set(expected), 'changed source set drift')
    return result


def apply_exceptions(deltas, exceptions, affected, changed, source_root, source_hashes):
    require(isinstance(exceptions, list), 'exception list malformed')
    remaining = {sha_bytes(json.dumps(d, sort_keys=True).encode()): d for d in deltas}
    require(len(remaining) == len(deltas), 'duplicate delta rows')
    for item in exceptions:
        require(set(item) == {'delta_sha256', 'object', 'field', 'identity',
                              'f', 'g', 'source_line', 'source_line_sha256', 'reason'},
                'exception row schema drift')
        digest = item['delta_sha256']
        require(digest in remaining and
                all(item[k] == remaining[digest][k]
                    for k in ('object', 'field', 'identity', 'f', 'g')),
                'exception does not match exact delta')
        require(item['object'] in affected and isinstance(item['source_line'], str)
                and re.fullmatch(r'[^:\n]+:[1-9][0-9]*', item['source_line'])
                and isinstance(item['reason'], str) and item['reason'].strip(),
                'exception missing affected source line/reason')
        changed_sources = affected[item['object']]['source_changed']
        file_name, number = item['source_line'].rsplit(':', 1)
        number = int(number)
        require(file_name in changed_sources and file_name in source_hashes and
                number in changed.get(file_name, set()),
                'exception source line outside changed closure')
        source_path = Path(source_root) / relative(file_name)
        within(source_path, Path(source_root))
        source_bytes = source_path.read_bytes()
        require(sha_bytes(source_bytes) == source_hashes[file_name],
                'exception source file content drift')
        lines = source_bytes.splitlines(keepends=True)
        require(number <= len(lines) and
                item['source_line_sha256'] == sha_bytes(lines[number - 1]),
                'exception changed line content drift')
        remaining.pop(digest)
    return list(remaining.values())


def result_status(remaining, archive_header_deltas):
    return 'PASS' if not remaining and not archive_header_deltas else 'HOLD'


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def check_limits():
    for kind, value in ((resource.RLIMIT_AS, AS_BYTES),
                        (resource.RLIMIT_CPU, CPU_SECONDS),
                        (resource.RLIMIT_FSIZE, FSIZE_BYTES),
                        (resource.RLIMIT_CORE, 0)):
        require(resource.getrlimit(kind) == (value, value), 'child resource limit drift')


def admit_child_product(side, kind, product_path, input_path, pairs, sides):
    require(side in ('fb', 'gb') and kind in ('object', 'archive'),
            'child side/kind not reviewed')
    selected = [row for row in pairs if row['path'] == product_path]
    require(len(selected) == 1 and selected[0]['kind'] == kind,
            'child path/kind not in reviewed pair map')
    products = [row for row in sides[side][0] if row['path'] == product_path]
    require(len(products) == 1, 'child path not in reviewed side metadata')
    product = products[0]
    expected_path = RUN[side] / 'build' / relative(product_path)
    require(input_path == expected_path, 'child input path is not canonical')
    within(input_path, RUN[side])
    seal = RUN[side] / 'SHA256SUMS-RUN'
    regular(seal)
    raw_seal = seal.read_bytes()
    require(sha_bytes(raw_seal) == ROOT_SHA[side], 'child RUN root drift')
    rows = seal_entries(seal, raw_seal)
    require(rows.get('build/' + product_path) == product['sha256'],
            'child RUN product hash map drift')
    st = regular(input_path)  # Metadata only: the product is not opened here.
    require(st.st_size == product['bytes'] and 0 < st.st_size <= 160 * 1024 * 1024,
            'child product size drift')
    return {**product, 'side': side, 'kind': kind}


def admit_child_receipt(side, product_path, receipt_path):
    require(receipt_path.is_absolute() and
            receipt_path.name == f'{side}-{product_path.replace("/", "_")}.receipt.json',
            'child receipt path is not canonical')
    output = receipt_path.parent
    require(output.parent == BASE and output.name.startswith(STEM + 'inventory-run-'),
            'child receipt outside owned output namespace')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    base_fd = os.open(BASE, flags)
    try:
        base_stat = os.fstat(base_fd)
        at_path = BASE.lstat()
        require((base_stat.st_dev, base_stat.st_ino) ==
                (at_path.st_dev, at_path.st_ino),
                'child receipt base directory drift')
        output_fd = os.open(output.name, flags, dir_fd=base_fd)
        try:
            st = os.fstat(output_fd)
            require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and
                    stat.S_IMODE(st.st_mode) == 0o700,
                    'child receipt output not fresh owned directory')
            check_receipt_directory(base_fd, output_fd, output.name)
            try:
                os.stat(receipt_path.name, dir_fd=output_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise Hold('child receipt output not fresh owned directory')
            return base_fd, output_fd
        except BaseException:
            os.close(output_fd)
            raise
    except BaseException:
        os.close(base_fd)
        raise


def check_receipt_directory(base_fd, output_fd, output_name):
    try:
        at_name = os.stat(output_name, dir_fd=base_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise Hold('child receipt output directory moved') from None
    pinned = os.fstat(output_fd)
    require(stat.S_ISDIR(at_name.st_mode) and
            (at_name.st_dev, at_name.st_ino) == (pinned.st_dev, pinned.st_ino),
            'child receipt output directory moved')


def write_child_receipt(receipt_fds, receipt_path, receipt):
    base_fd, output_fd = receipt_fds
    check_receipt_directory(base_fd, output_fd, receipt_path.parent.name)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(receipt_path.name, flags, 0o600, dir_fd=output_fd)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + '\n')
    try:
        check_receipt_directory(base_fd, output_fd, receipt_path.parent.name)
    except BaseException:
        os.unlink(receipt_path.name, dir_fd=output_fd)
        raise


def child_parse(kind, input_path, receipt_path, product, receipt_fds):
    child_limits()
    check_limits()
    parser = load_pinned(PARSER / 'INVENTORY.py', 'SEALED_INVENTORY',
                         '4944bc67066bf55c26e9da4a3fa075458b4c5f783164a88e0730d5ed9f6f23a4')
    check_imports()
    regular(input_path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(input_path, flags)
    with os.fdopen(fd, 'rb') as f:
        st = os.fstat(f.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and
                st.st_size == product['bytes'], 'child input not reviewed regular size')
        raw = f.read(parser.MAX_INPUT + 1)
    require(len(raw) <= parser.MAX_INPUT and len(raw) == product['bytes'],
            'child input cap or size drift')
    require(sha_bytes(raw) == product['sha256'], 'child input hash drift before parse')
    result = parser.parse_elf(raw) if kind == 'object' else parser.parse_archive(raw)
    json.dump(result, sys.stdout, sort_keys=True, separators=(',', ':'))
    sys.stdout.write('\n')
    sys.stdout.flush()
    check_imports()
    check_limits()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(isinstance(peak, int) and 0 < peak <= AS_BYTES // 1024,
            'child peak RSS invalid')
    receipt = {'side': product['side'], 'path': product['path'],
               'bytes': len(raw), 'input_sha256': sha_bytes(raw), 'kind': kind,
               'peak_rss_kib': peak,
               'parser_sha256': '4944bc67066bf55c26e9da4a3fa075458b4c5f783164a88e0730d5ed9f6f23a4'}
    write_child_receipt(receipt_fds, receipt_path, receipt)


def run_child(side, product, output, release_path, binding_sha):
    input_path = RUN[side] / 'build' / relative(product['path'])
    name = f'{side}-{product["path"].replace("/", "_")}'
    stdout_path, stderr_path, receipt_path = (output / (name + suffix)
                                            for suffix in ('.json', '.stderr', '.receipt.json'))
    require(not any(p.exists() for p in (stdout_path, stderr_path, receipt_path)),
            'child output collision')
    start = time.monotonic()
    timed_out, spawn_error = False, None
    with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
        argv = ['/usr/bin/python3', '-I', '-S', str(Path(__file__).resolve()), '--child',
                side, product['kind'], product['path'], str(input_path),
                str(receipt_path), str(release_path), binding_sha, '3']
        try:
            proc = subprocess.run(argv, stdout=out, stderr=err, timeout=WALL_SECONDS,
                                  cwd='/', env={'LC_ALL': 'C', 'LANG': 'C', 'PATH': '/usr/bin:/bin',
                                                'HOME': '/nonexistent', 'PYTHONNOUSERSITE': '1'},
                                  preexec_fn=child_limits, pass_fds=(3,), check=False)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = None
        except BaseException as exc:
            spawn_error = type(exc).__name__
            proc = None
    code = None if proc is None else proc.returncode
    status = ('TIMEOUT' if timed_out else 'SPAWN_ERROR' if spawn_error else
              'SIGNAL' if isinstance(code, int) and code < 0
              else 'NONZERO' if code != 0 else 'EXIT')
    receipt = {'side': side, 'path': product['path'], 'status': status, 'exit_code': code,
               'signal': -code if isinstance(code, int) and code < 0 else None,
               'external_binding_sha256': binding_sha,
               'wall_seconds': time.monotonic() - start, 'stdout_sha256': sha_file(stdout_path),
               'stdout_bytes': stdout_path.stat().st_size,
               'stderr_sha256': sha_file(stderr_path), 'stderr_bytes': stderr_path.stat().st_size,
               'limits': {'address_space_bytes': AS_BYTES, 'cpu_seconds': CPU_SECONDS,
                          'wall_seconds': WALL_SECONDS, 'file_size_bytes': FSIZE_BYTES}}
    if spawn_error:
        receipt['error_type'] = spawn_error
    if receipt_path.is_file():
        receipt['child_receipt_sha256'] = sha_file(receipt_path)
    if status == 'EXIT' and receipt['stderr_bytes']:
        receipt['status'] = 'STDERR'
    elif status == 'EXIT' and not receipt_path.is_file():
        receipt['status'] = 'MISSING_RECEIPT'
    (output / (name + '.child.json')).write_text(json.dumps(receipt, sort_keys=True) + '\n')
    require(not timed_out and code == 0 and receipt['stderr_bytes'] == 0 and
            receipt_path.is_file(),
            'child failed or emitted stderr')
    try:
        inner = json.loads(receipt_path.read_text())
        require(set(inner) == {'side', 'path', 'bytes', 'input_sha256', 'kind',
                               'parser_sha256', 'peak_rss_kib'}
                and inner['side'] == side and inner['path'] == product['path']
                and inner['bytes'] == product['bytes']
                and inner['input_sha256'] == product['sha256'] and inner['kind'] == product['kind']
                and inner['parser_sha256'] ==
                '4944bc67066bf55c26e9da4a3fa075458b4c5f783164a88e0730d5ed9f6f23a4',
                'child input receipt drift')
        require(isinstance(inner['peak_rss_kib'], int) and
                0 < inner['peak_rss_kib'] <= AS_BYTES // 1024, 'child RSS receipt invalid')
        receipt['peak_rss_kib'] = inner['peak_rss_kib']
        result = json.loads(stdout_path.read_text())
        require(isinstance(result, dict), 'child output schema drift')
    except BaseException as exc:
        receipt.update(status='INVALID_OUTPUT', error_type=type(exc).__name__)
        (output / (name + '.child.json')).write_text(json.dumps(receipt, sort_keys=True) + '\n')
        raise
    (output / (name + '.child.json')).write_text(json.dumps(receipt, sort_keys=True) + '\n')
    return result, receipt


def write_result_and_parent_completion(output, result):
    result_path = output / 'RESULT.json'
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    check_imports()
    require(resource.getrlimit(resource.RLIMIT_AS) == (AS_BYTES, AS_BYTES),
            'parent address-space limit drift after result')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(isinstance(peak, int) and 0 < peak <= AS_BYTES // 1024,
            'parent peak RSS invalid after result')
    (output / 'PARENT-COMPLETED.json').write_text(json.dumps({
        'result_sha256': sha_file(result_path), 'parent_peak_rss_kib': peak,
        'address_space_bytes': AS_BYTES, 'status': 'COMPLETE'}, sort_keys=True) + '\n')


def output_identity(output, fresh):
    require(output.is_absolute() and output.parent == BASE and
            output.name.startswith(STEM + 'inventory-run-'),
            'output outside launcher-created namespace')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    base_fd = os.open(BASE, flags)
    try:
        output_fd = os.open(output.name, flags, dir_fd=base_fd)
        try:
            st = os.fstat(output_fd)
            at_name = os.stat(output.name, dir_fd=base_fd, follow_symlinks=False)
            require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and
                    st.st_gid == os.getgid() and stat.S_IMODE(st.st_mode) == 0o700 and
                    (st.st_dev, st.st_ino) == (at_name.st_dev, at_name.st_ino) and
                    (not fresh or not os.listdir(output_fd)),
                    'launcher output identity or freshness drift')
            return st.st_dev, st.st_ino
        finally:
            os.close(output_fd)
    finally:
        os.close(base_fd)


def precreated_output_identity(output):
    return output_identity(output, True)


def launcher_handoff_fields(binding_sha, ledger_fd):
    """Read the root-created immutable authority; argv digest is not authority."""
    require(ledger_fd >= 3 and os.getuid() == os.geteuid() != 0,
            'launcher handoff privilege or descriptor drift')
    st = os.fstat(ledger_fd)
    require(protected_handoff_stat(st), 'launcher handoff is not protected')
    seals = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
             fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
    require(fcntl.fcntl(ledger_fd, fcntl.F_GET_SEALS) & seals == seals,
            'launcher handoff is not sealed')
    raw = os.pread(ledger_fd, 513, 0)
    fields = raw.decode('ascii', 'strict').splitlines()
    require(len(raw) <= 512 and raw.endswith(b'\n') and len(fields) == 8 and
            fields[0] == 'eshkol-off-inventory-handoff-v2' and
            fields[1] == binding_sha and fields[2] == str(os.getuid()) and
            re.fullmatch(r'[0-9a-f]{32}', fields[7]) is not None,
            'launcher handoff record drift or replay')
    return fields


def consume_launcher_handoff(output, binding_sha, ledger_fd):
    """Accept the one-use launcher record and retain FD3 for admitted children."""
    fields = launcher_handoff_fields(binding_sha, ledger_fd)
    require(fields[3] == str(os.getpid()) and fields[4] == process_starttime(),
            'launcher handoff record drift or replay')
    output_identity = precreated_output_identity(output)
    require(fields[5:7] == [str(x) for x in output_identity],
            'launcher output inode drift')
    return output_identity


def consume_child_handoff(output, binding_sha, ledger_fd):
    """A parser child must be a live child of the authorized run process."""
    fields = launcher_handoff_fields(binding_sha, ledger_fd)
    parent_pid = os.getppid()
    require(fields[3] == str(parent_pid) and fields[4] == process_starttime(parent_pid),
            'child handoff parent drift or replay')
    actual = output_identity(output, False)
    require(fields[5:7] == [str(x) for x in actual],
            'child handoff output inode drift')
    os.close(ledger_fd)
    return actual


def process_starttime(pid=None):
    selected = '/proc/self/stat' if pid is None else f'/proc/{pid}/stat'
    tail = Path(selected).read_text().rsplit(')', 1)[1].split()
    require(len(tail) >= 20 and tail[19].isdigit(), 'process start identity unavailable')
    return tail[19]


def protected_handoff_stat(st):
    return (stat.S_ISREG(st.st_mode) and st.st_uid == 0 and st.st_nlink == 0 and
            stat.S_IMODE(st.st_mode) == 0o600 and st.st_size <= 512)


def run(release_path, output, binding_sha, output_identity):
    release = review_gate(release_path, binding_sha)
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    require(precreated_output_identity(output) == output_identity,
            'launcher output changed after handoff')
    try:
        plan, pairs, closure, sides = authenticate_metadata()
        verify_tools()
        check_imports()
        (output / 'ADMISSION.json').write_text(json.dumps({
            'release_sha256': sha_file(release_path),
            'external_binding_sha256': binding_sha,
            'runner_seal_sha256': release['runner_manifest_sha256'],
            'input_seals': {str(k): v for k, v in PINS.items()},
            'run_seals': ROOT_SHA, 'pair_map_sha256': sha_file(PLAN / 'PAIR-MAP.json'),
            'source_closure_sha256': sha_file(PLAN / 'SOURCE-CLOSURE.json'),
            'parser_sha256': sha_file(PARSER / 'INVENTORY.py'),
            'limit': {'address_space_bytes': AS_BYTES, 'cpu_seconds': CPU_SECONDS,
                      'wall_seconds': WALL_SECONDS, 'file_size_bytes': FSIZE_BYTES},
        }, sort_keys=True, indent=2) + '\n')
        manifest_counts = {s: authenticate_run_files(s) for s in ('fb', 'gb')}
        exceptions_path = Path(release_path).parent / 'EXCEPTIONS.json'
        regular(exceptions_path)
        exception_raw = exceptions_path.read_bytes()
        require(sha_bytes(exception_raw) == release['exception_file_sha256'],
                'exception file hash drift')
        exceptions = json.loads(exception_raw)
        source = sys.modules['SOURCE_CLOSURE']
        old, new = plan.HEAD['gb'][0], plan.HEAD['fb'][0]
        changed = changed_lines(source.git('diff', '--unified=0', old, new),
                                {row['path'] for row in closure['source_delta']['files']})
        products = {s: {p['path']: p for p in sides[s][0]} for s in ('fb', 'gb')}
        affected = {r['object']: r for r in closure['object_closure']}
        deltas, receipts, archives, archive_header_deltas = [], [], {}, []
        for pair in pairs:
            path = pair['path']
            parsed = {}
            for side in ('fb', 'gb'):
                p = {**products[side][path], 'kind': pair['kind']}
                parsed[side], rec = run_child(side, p, output, release_path, binding_sha)
                receipts.append(rec)
                require(parsed[side]['sha256'] == p['sha256'], 'parser raw hash drift')
            if pair['kind'] == 'archive':
                for side in ('fb', 'gb'):
                    command = read_run_json(side,
                                            f'children/command-{pair["index"]:03}.json')['inner_argv']
                    order = archive_order(command, path, products[side])
                    archives[side, path] = index_members(parsed[side], order, products[side])
                    for member, object_path in zip(parsed[side]['members'], order):
                        standalone_path = output / f'{side}-{object_path.replace("/", "_")}.json'
                        require(standalone_path.is_file() and
                                member['elf'] == json.loads(standalone_path.read_text()),
                                'archive member ELF inventory differs from standalone')
                require(archives['fb', path]['index'] == archives['gb', path]['index'],
                        'F/G archive index drift')
                require(archives['fb', path]['member_names'] == archives['gb', path]['member_names'],
                        'F/G archive member list drift')
                (output / f'archive-{pair["index"]:03}.json').write_text(
                    json.dumps({'path': path, 'f': archives['fb', path],
                                'g': archives['gb', path]}, sort_keys=True) + '\n')
                require(len(parsed['fb']['headers']) == len(parsed['gb']['headers']),
                        'F/G archive header count drift')
                for ordinal, (fh, gh) in enumerate(zip(parsed['fb']['headers'],
                                                        parsed['gb']['headers'])):
                    require(fh['kind'] == gh['kind'] and fh['raw_name_hex'] == gh['raw_name_hex'],
                            'F/G archive header identity drift')
                    if fh != gh:
                        archive_header_deltas.append({'archive': path, 'ordinal': ordinal,
                                                      'changed_fields': sorted(k for k in fh
                                                                               if fh[k] != gh[k]),
                                                      'f': fh, 'g': gh})
            else:
                froot = [(plan.SOURCE['fb'], '<SOURCE>'),
                         (str(BASE / f'{STEM}config-run-20261011-fb'), '<CONFIG>'),
                         (str(RUN['fb']), '<RUN>')]
                groot = [(plan.SOURCE['gb'], '<SOURCE>'),
                         (str(BASE / f'{STEM}config-run-20261011-gb'), '<CONFIG>'),
                         (str(RUN['gb']), '<RUN>')]
                # Debug normalization reads only already authenticated bytes.
                raw = {}
                for side in ('fb', 'gb'):
                    raw[side] = (RUN[side] / 'build' / relative(path)).read_bytes()
                    require(sha_bytes(raw[side]) == products[side][path]['sha256'],
                            'normalized raw product changed after child')
                f = canonical_elf(parsed['fb'], raw['fb'], froot)
                g = canonical_elf(parsed['gb'], raw['gb'], groot)
                deltas += delta_rows(path, f, g, next(x for x in closure['object_closure']
                                                      if x['object'] == path))
            (output / f'comparison-{pair["index"]:03}.json').write_text(
                json.dumps({'path': path, 'delta_count': len(deltas)}, sort_keys=True) + '\n')
        source_hashes = {row['path']: row['fb_sha256']
                         for row in closure['source_delta']['files']}
        remaining = apply_exceptions(deltas, exceptions, affected, changed,
                                     plan.SOURCE['fb'], source_hashes)
        result = {'status': result_status(remaining, archive_header_deltas),
                  'manifest_counts': manifest_counts,
                  'pairs': len(pairs), 'deltas': deltas, 'unreviewed': remaining,
                  'archive_header_deltas': archive_header_deltas,
                  'child_receipts': receipts}
        write_result_and_parent_completion(output, result)
    except BaseException as exc:
        (output / 'FAILED.json').write_text(json.dumps({'status': 'HOLD',
                            'error_type': type(exc).__name__, 'error': str(exc)}, sort_keys=True) + '\n')
        raise
    finally:
        files = sorted(p for p in output.iterdir() if p.is_file() and
                       p.name not in ('SHA256SUMS-RUN', 'COMPLETED.json'))
        manifest = output / 'SHA256SUMS-RUN'
        with manifest.open('x') as stream:
            for p in files:
                stream.write(f'{sha_file(p)}  {p.name}\n')
        status = ('HOLD' if (output / 'FAILED.json').exists() else
                  json.loads((output / 'RESULT.json').read_text())['status'])
        (output / 'COMPLETED.json').write_text(json.dumps({
            'status': status, 'manifest_sha256': sha_file(manifest),
            'result_sha256': sha_file(output / 'RESULT.json') if
            (output / 'RESULT.json').exists() else None}, sort_keys=True) + '\n')
        for p in output.iterdir():
            if p.is_file():
                p.chmod(0o444)
        output.chmod(0o555)


def main(argv):
    if len(argv) == 10 and argv[1] == '--child':
        require(argv[9] == '3', 'child descriptor argument drift')
        consume_child_handoff(Path(argv[6]).parent, argv[8], 3)
        review_gate(Path(argv[7]), argv[8])
        _, pairs, _, sides = authenticate_metadata()
        verify_tools()
        product = admit_child_product(argv[2], argv[3], argv[4], Path(argv[5]),
                                      pairs, sides)
        receipt_fds = admit_child_receipt(argv[2], argv[4], Path(argv[6]))
        try:
            child_parse(argv[3], Path(argv[5]), Path(argv[6]), product, receipt_fds)
        finally:
            for fd in receipt_fds:
                os.close(fd)
    elif len(argv) == 6 and argv[1] == '--run':
        require(argv[5] == '3', 'launcher descriptor argument drift')
        output = Path(argv[3])
        identity = consume_launcher_handoff(output, argv[4], 3)
        try:
            run(Path(argv[2]), output, argv[4], identity)
        finally:
            os.close(3)
    else:
        raise Hold('usage: launcher-only RUNNER.py --run RELEASE.json '
                   'PRECREATED_OUTPUT_DIR BINDING_SHA256 LEDGER_FD')


if __name__ == '__main__':
    try:
        main(sys.argv)
    except BaseException as exc:
        print(f'HOLD: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
