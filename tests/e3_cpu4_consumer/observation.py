"""Strict actual six-word publication transcript admission; no math-oracle claim."""
import json
import math
import re
import struct
import sys
from pathlib import Path

PATTERN=re.compile(r'CPU4-CONSUMER-REPORT #\((\d+) (\d+) (\d+) (\d+) (\d+) (\d+)\)\nCPU4-CONSUMER-PASS checks=([1-9]\d*)\n\Z')

def parse(text):
    match=PATTERN.fullmatch(text)
    if not match:raise ValueError('unexpected/missing consumer transcript')
    words=tuple(map(int,match.groups()[:6]))
    if any(word>0xffffffff for word in words[:4]) or words[4:]!=(8,2):
        raise ValueError('invalid metric words/physical counts')
    values=[struct.unpack('<f',struct.pack('<I',word))[0] for word in words[:4]]
    mean,weight,perplexity,accuracy=values
    if not all(math.isfinite(v) for v in values) or weight!=8 or mean<0 or perplexity<1 or not 0<=accuracy<=1:
        raise ValueError('invalid finite evaluation result')
    return {'metric_u32':words[:4],'physical_tokens':words[4],'batches':words[5],
            'checks':int(match.group(7)),'scope':'candidate-off direct consumer, not candidate-on compatibility'}

if __name__=='__main__':
    print(json.dumps(parse(Path(sys.argv[1]).read_text()),sort_keys=True))
