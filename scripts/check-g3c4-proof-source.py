#!/usr/bin/env python3
"""Conservative shell-site audit for the nine terminal-close runner scripts."""
from pathlib import Path
import re
import sys

NAMES = (
    "test-g3c4-p2g2-terminal-close.sh",
    "test-g3c4-p2g2-t1-prefix-coordinator.sh",
    "test-g3c4-p2g1-pending.sh",
    "test-g3c4-p2g2-protected-scope.sh",
    "test-g3c4-p2g2-carrier-bridge.sh",
    "test-g3c4-p2g2-prefix-commit.sh",
    "test-g3c4-p2g2-first-frame.sh",
    "test-g3c4-p2g2-terminal-ids-clone.sh",
    "test-g3c4-p2g2-terminal.sh",
)
EXTERNAL = r'(?:python3|bash|env|ar|nm|awk|grep|comm|cmp|sha256sum|sort|cp|cat|git|mkdir|chmod|mktemp|"\$(?:cc|cxx|runner|eshkol_runner)"|"\$\{runtime\[@\]\}"|"\$(?:tmp|directory)/[^" ]+")'
START = re.compile(rf'(?:^|[;&|])\s*(?:if\s+|!\s+|LC_ALL=C\s+|[A-Z_][A-Z_0-9]*=\S+\s+)*({EXTERNAL})(?=\s|$)')
COMMON_EXTERNAL = re.compile(
    r'(?<!common_probe )\b(?:awk|sed|sort|head|uname|git|readlink|sha256sum|cmp|comm)\s+\S')


def segments(line: str) -> list[str]:
    result = []
    quote = None
    escaped = False
    start = 0
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == '\\' and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in '|;&':
            result.append(line[start:index])
            start = index + 1
    result.append(line[start:])
    return result


def scan_text(name: str, source: str) -> list[str]:
    findings = []
    heredoc = False
    continued = False
    for line_number, line in enumerate(source.splitlines(), 1):
        if '<<WRAPPER' in line:
            heredoc = True
        if heredoc:
            if line == 'WRAPPER':
                heredoc = False
            continue
        if '$(mktemp ' in line:
            findings.append(f'{name}:{line_number}: unwrapped mktemp')
        parts = segments(line)
        for index, part in enumerate(parts):
            if continued and index == 0:
                continue
            part = re.sub(r'^\s*(?:if\s+|!\s+|LC_ALL=C\s+|[A-Z_][A-Z_0-9]*=\S+\s+)*', '', part)
            if re.match(r'^proof_command\s+', part):
                continue
            if START.search(part):
                findings.append(f'{name}:{line_number}: unwrapped external command')
        continued = line.rstrip().endswith('\\') and not re.search(
            r'(?:\|\||&&|[|;])\s*\\$', line)
    return findings


def audit(root: Path) -> list[str]:
    findings = [finding for name in NAMES
                for finding in scan_text(name, (root / 'scripts' / name).read_text())]
    for number, line in enumerate((root / 'scripts/common.sh').read_text().splitlines(), 1):
        if line.lstrip().startswith('#'):
            continue
        if COMMON_EXTERNAL.search(line):
            findings.append(f'common.sh:{number}: unwrapped external probe')
    return findings


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    findings = audit(root)
    if findings:
        print('\n'.join(findings), file=sys.stderr)
        raise SystemExit(86)
    print(f'G3-C4 shell external site audit PASS: scripts={len(NAMES)}')
