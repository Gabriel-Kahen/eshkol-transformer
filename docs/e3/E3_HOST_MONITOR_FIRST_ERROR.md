# E3 one-use host monitor first-error candidate

The sealed L run's `bounded-run.py` remains immutable. Its live guest-output
scan failed with `unsafe-guest-output` and stopped the Docker client, but the
monitor row discarded the first `OSError` or `ValueError`. A later clean final
scan cannot identify a transient offending entry. The source-only successor
candidate is [`tests/e3_host_monitor/bounded_run.py`](../../tests/e3_host_monitor/bounded_run.py),
copied from L's bounded runner and changed only to retain a safe first fault.
It is not part of an executed one-use launcher or an E3 acceptance result.

The monitor retains the first live scan fault, or the final scan fault if no
live fault occurred, in six added `docker.monitor.tsv` fields:
`first_output_error_phase`, `class`, `code`, `path`, `message`, and `errno`
(each prefixed `first_output_error_`). A clean scan writes `none` in each.
Codes distinguish traversal bound, directory scan/stat error, symlink,
hardlinked regular file, and other nonregular entry. The message is fixed by
code; raw exception text, symlink targets, and absolute host paths are never
written. The path is built only from traversed guest entry names, percent-
encodes bytes outside URL-unreserved characters, rejects `.`/`..` components,
and is capped at 256 characters. Errno is bounded to four digits. A transient
entry's first error survives a clean final scan.

The prior 50,000-entry, 32-level, two-second traversal bounds, stream/guest-
output/wall caps, no-follow stat, group termination, final scan, and exit-74
failure policy remain. Offline tests cover a transient symlink removed before
the final scan, a guest-relative control-character path, OS errors with secret
text/host paths, hardlink/FIFO refusal, bounds, successful completion, and
stream/guest-output/wall caps. The future one-use recipe must copy and pin this
reviewed candidate as a new component; it must not edit or replay L. Independent
review and an authorized fresh one-use run remain separate gates.
