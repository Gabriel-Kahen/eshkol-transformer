#!/usr/bin/env python3
"""Development-only elapsed/child-process peak RSS evidence for one command."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

started = time.monotonic()
result = subprocess.run(sys.argv[2:], check=False)
Path(sys.argv[1]).write_text(json.dumps({
    'elapsed_seconds': round(time.monotonic() - started, 3),
    'child_max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    'returncode': result.returncode,
}, sort_keys=True) + '\n')
sys.exit(result.returncode)
