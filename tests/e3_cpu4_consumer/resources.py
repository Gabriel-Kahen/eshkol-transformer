"""Development-only genuine V256 raw-byte corpus, independent of native math."""
from pathlib import Path
import json
import sys
from tests.d2.test_resources import write_d1_resource
from tests.t1.reference import Identity, fingerprint, encode_artifact

TOKENS = (3, 197, 3, 41, 7, 2, 9, 5, 1, 11, 17, 23, 29)

def generate(root):
    root.mkdir()
    identity = Identity()
    assert not identity.specials and not identity.prefix and not identity.suffix
    (root / 'tokenizer.tsv').write_bytes(encode_artifact(identity))
    corpus = root / 'corpus'
    corpus.mkdir()
    write_d1_resource(corpus, (TOKENS,), fingerprint=fingerprint(identity), vocab=256)
    (root / 'reference.json').write_text(json.dumps({
        'vocabulary': 256, 'fingerprint': fingerprint(identity), 'tokens': TOKENS,
        'skipped_batches': 1, 'remaining_batches': 2, 'remaining_tokens': 8,
        'metric_oracle': 'Accepted native frame oracle; consumer checks exact repeat bytes, not a new numerical oracle.'
    }, indent=2) + '\n')

if __name__ == '__main__':
    generate(Path(sys.argv[1]))
