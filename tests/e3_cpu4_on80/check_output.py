"""Exact old six words and assertion totals, with honest current-tuple scope."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.e3_cpu4_consumer.observation import parse

WORDS = (1085385783, 1090519040, 1132521808, 0)
CHECKS = {'absent': 217, 'present': 219}


def check(text, mode):
    if mode not in CHECKS:
        raise ValueError('unknown gradient state')
    value = parse(text)
    if tuple(value['metric_u32']) != WORDS or value['checks'] != CHECKS[mode]:
        raise ValueError('original publication words/assertion total changed')
    value['scope'] = 'candidate-ON80 current-root consumer; normal once; no SAN/retention claim'
    return value


if __name__ == '__main__':
    print(json.dumps(check(Path(sys.argv[1]).read_text(), sys.argv[2]), sort_keys=True))
