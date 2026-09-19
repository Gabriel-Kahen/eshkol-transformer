#!/usr/bin/env python3
"""Emit and select exact-tree CI evidence without trusting executable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, NoReturn
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "CI_EVIDENCE_V1="
MAX_COMMAND_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024
COMMAND_TIMEOUT_SECONDS = 20
FINAL_AGGREGATION = "Ubuntu 22.04 x86-64 / Clang-LLVM 21.1.8"
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
CI_EVENTS = {"push", "pull_request", "merge_group", "workflow_dispatch"}
SUITES = (
    "canonical-build",
    "native-numerics",
    "native-optimizer",
    "diagnostic-transport",
    "model-composition",
    "contracts-data",
    "checkpoint-io",
    "parameter-state",
    "byte-tokenizer",
    "bpe-tokenizer",
    "bpe-boundary",
    "shard-loader",
    "c2-format",
    "c2-state",
    "c2-save",
    "c2-load",
    "c2-public",
    "c2-operational",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
JOB_LOG_PATH = re.compile(
    r"/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/jobs/[1-9][0-9]*/logs\Z"
)


class EvidenceError(RuntimeError):
    """Evidence is missing, malformed, stale, or otherwise untrusted."""


class PendingRunError(EvidenceError):
    """The newest exact-commit CI run is still active."""


@dataclass(frozen=True)
class Selection:
    reused: bool
    reason: str
    run_id: int | None = None
    docs_verified: bool = False


Api = Callable[[str], str | bytes]


def _fail(message: str) -> NoReturn:
    raise EvidenceError(message)


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        _fail(f"{field} must be a lowercase 40-character commit ID")
    return value


def _repository(value: object) -> str:
    if not isinstance(value, str) or not REPOSITORY.fullmatch(value):
        _fail("repository must be owner/name")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str, description: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise EvidenceError(f"malformed {description}: {error}") from error


def run_bounded(command: list[str]) -> str:
    """Run a fixed local command with bounded time and retained output."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise EvidenceError(f"command timed out: {command[0]}") from error
        stdout_size = stdout.tell()
        stderr_size = stderr.tell()
        if stdout_size > MAX_COMMAND_BYTES or stderr_size > MAX_COMMAND_BYTES:
            raise EvidenceError(f"command output exceeded limit: {command[0]}")
        stdout.seek(0)
        stderr.seek(0)
        try:
            output = stdout.read().decode("utf-8")
            diagnostic = stderr.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"command emitted non-UTF-8 output: {command[0]}") from error
    if result.returncode != 0:
        detail = diagnostic.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise EvidenceError(f"command failed ({result.returncode}): {command[0]}{suffix}")
    return output


@lru_cache(maxsize=1)
def _gh_allows_escape_sequences() -> bool:
    return "--allow-escape-sequences" in run_bounded(["gh", "api", "--help"])


def gh_api(path: str) -> str:
    if not path.startswith("/repos/") or ".." in path:
        _fail("refusing non-repository GitHub API path")
    log_path = JOB_LOG_PATH.fullmatch(path) is not None
    if "/actions/jobs/" in path and path.endswith("/logs") and not log_path:
        _fail("refusing malformed GitHub job log path")
    command = ["gh", "api"]
    if log_path and _gh_allows_escape_sequences():
        command.append("--allow-escape-sequences")
    command.append(path)
    return run_bounded(command)


def _api_text(api: Api, path: str, description: str) -> str:
    try:
        response = api(path)
    except EvidenceError:
        raise
    except Exception as error:
        raise EvidenceError(f"GitHub API failed for {description}: {error}") from error
    encoded = response.encode("utf-8") if isinstance(response, str) else response
    if not isinstance(encoded, bytes):
        _fail(f"GitHub API returned an invalid {description} response")
    if len(encoded) > MAX_COMMAND_BYTES:
        _fail(f"GitHub API {description} response exceeded limit")
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"GitHub API returned non-UTF-8 {description}") from error


def _api_json(api: Api, path: str, description: str) -> object:
    return strict_json(_api_text(api, path, description), description)


def _local_revision(name: str) -> str:
    return run_bounded(["git", "rev-parse", name]).strip()


def make_report(env: dict[str, str] | None = None) -> dict[str, object]:
    values = os.environ if env is None else env
    repository = _repository(values.get("GITHUB_REPOSITORY"))
    run_id = _positive_int_from_env(values, "GITHUB_RUN_ID")
    attempt = _positive_int_from_env(values, "GITHUB_RUN_ATTEMPT")
    commit = _sha(_local_revision("HEAD"), "HEAD")
    tree = _sha(_local_revision("HEAD^{tree}"), "HEAD tree")
    if values.get("GITHUB_SHA") != commit:
        _fail("checked-out HEAD does not match GITHUB_SHA")
    if values.get("GITHUB_JOB") != "evidence":
        _fail("evidence may only be emitted by the evidence job")
    return {
        "version": 1,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": attempt,
        "commit": commit,
        "tree": tree,
        "suites": list(SUITES),
    }


def _positive_int_from_env(env: dict[str, str], name: str) -> int:
    value = env.get(name, "")
    if not value.isdigit():
        _fail(f"{name} must be a positive integer")
    return _positive_int(int(value), name)


def emit(env: dict[str, str] | None = None) -> None:
    report = make_report(env)
    print(PREFIX + json.dumps(report, sort_keys=True, separators=(",", ":")))


def _dict(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{description} must be a JSON object")
    return value


def _list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{description} must be a JSON array")
    return value


def _latest_run(
    payload: object,
    repository: str,
    commit: str,
    required_event: str | None = None,
) -> dict[str, object] | None:
    body = _dict(payload, "workflow-runs response")
    runs = _list(body.get("workflow_runs"), "workflow_runs")
    total = body.get("total_count")
    if type(total) is not int or total != len(runs) or total > 100:
        _fail("workflow-runs response is incomplete or exceeds 100 runs")
    matching: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for raw in runs:
        run = _dict(raw, "workflow run")
        run_id = _positive_int(run.get("id"), "workflow run id")
        if run_id in seen_ids:
            _fail("workflow-runs response contains a duplicate run")
        seen_ids.add(run_id)
        if run.get("head_sha") != commit:
            continue
        repo = _dict(run.get("repository"), "workflow run repository")
        if repo.get("full_name") != repository:
            _fail("workflow run repository does not match the requested repository")
        workflow_path = run.get("path")
        if (
            not isinstance(workflow_path, str)
            or workflow_path.split("@", 1)[0] != CI_WORKFLOW_PATH
        ):
            _fail("workflow run path is not the supported CI workflow")
        event = run.get("event")
        if event not in CI_EVENTS:
            _fail("workflow run event is not supported for evidence reuse")
        if required_event is not None and event != required_event:
            continue
        created = run.get("created_at")
        if not isinstance(created, str) or not TIMESTAMP.fullmatch(created):
            _fail("workflow run created_at is invalid")
        try:
            datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError as error:
            raise EvidenceError("workflow run created_at is invalid") from error
        matching.append(run)
    if not matching:
        return None
    return max(
        matching,
        key=lambda run: (
            datetime.fromisoformat(str(run["created_at"]).replace("Z", "+00:00")),
            int(run["id"]),
        ),
    )


def _name_matches(name: object, suffix: str) -> bool:
    return isinstance(name, str) and (name == suffix or name.endswith(" / " + suffix))


def _validated_jobs(payload: object, attempt: int) -> tuple[list[dict[str, object]], int]:
    body = _dict(payload, "jobs response")
    jobs_raw = _list(body.get("jobs"), "jobs")
    total = body.get("total_count")
    if type(total) is not int or total != len(jobs_raw) or total > 100:
        _fail("exact-attempt jobs response is incomplete or exceeds 100 jobs")
    jobs: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for raw in jobs_raw:
        job = _dict(raw, "job")
        job_id = _positive_int(job.get("id"), "job id")
        if job_id in seen_ids:
            _fail("jobs response contains a duplicate job")
        seen_ids.add(job_id)
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            _fail(f"job is not completed successfully: {job.get('name', '<unnamed>')}")
        if "run_attempt" in job and job["run_attempt"] != attempt:
            _fail("job run_attempt does not match the selected attempt")
        name = job.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            _fail("jobs response contains a missing or duplicate job name")
        seen_names.add(name)
        jobs.append(job)

    expected_suffixes = {f"Supported suite / {suite}" for suite in SUITES}
    observed: dict[str, int] = {suffix: 0 for suffix in expected_suffixes}
    for job in jobs:
        name = job.get("name")
        matches = [suffix for suffix in expected_suffixes if _name_matches(name, suffix)]
        if isinstance(name, str) and "Supported suite / " in name and not matches:
            _fail(f"unexpected supported-suite job: {name}")
        for suffix in matches:
            observed[suffix] += 1
    missing_or_duplicate = [suffix for suffix, count in observed.items() if count != 1]
    if missing_or_duplicate:
        _fail("suite jobs are missing or duplicated")

    evidence_jobs = [job for job in jobs if _name_matches(job.get("name"), "Suite evidence")]
    aggregate_jobs = [job for job in jobs if _name_matches(job.get("name"), FINAL_AGGREGATION)]
    if len(evidence_jobs) != 1 or len(aggregate_jobs) != 1:
        _fail("final evidence or CI aggregation job is missing or duplicated")
    return jobs, int(evidence_jobs[0]["id"])


def _has_explicit_full_matrix_skips(
    candidate: dict[str, object],
    repository: str,
    api: Api,
) -> bool:
    run_id = _positive_int(candidate.get("id"), "workflow run id")
    attempt = _positive_int(candidate.get("run_attempt"), "workflow run attempt")
    encoded_repo = quote(repository, safe="/")
    jobs_path = (
        f"/repos/{encoded_repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        "?per_page=100"
    )
    body = _dict(_api_json(api, jobs_path, "exact-attempt jobs"), "jobs response")
    raw_jobs = _list(body.get("jobs"), "jobs")
    total = body.get("total_count")
    if type(total) is not int or total != len(raw_jobs) or total > 100:
        _fail("collapsed skipped-run jobs are incomplete or exceed 100 jobs")
    collapsed: dict[str, dict[str, object]] = {}
    seen_ids: set[int] = set()
    for raw in raw_jobs:
        job = _dict(raw, "job")
        job_id = _positive_int(job.get("id"), "job id")
        name = job.get("name")
        if (
            job_id in seen_ids
            or not isinstance(name, str)
            or not name
            or name in collapsed
            or job.get("status") != "completed"
            or job.get("conclusion") not in {"success", "skipped"}
            or ("run_attempt" in job and job["run_attempt"] != attempt)
        ):
            _fail("collapsed skipped-run jobs are malformed")
        seen_ids.add(job_id)
        collapsed[name] = job
    if set(collapsed) != {"CI topology", "Full suites", FINAL_AGGREGATION}:
        _fail("collapsed skipped-run job inventory is not exact")
    return (
        collapsed["CI topology"].get("conclusion") == "success"
        and collapsed["Full suites"].get("conclusion") == "skipped"
        and collapsed[FINAL_AGGREGATION].get("conclusion") == "success"
    )


def _parse_report(logs: str) -> dict[str, object]:
    if logs.count(PREFIX) != 1:
        _fail("evidence job log must contain exactly one evidence report")
    tail = logs.split(PREFIX, 1)[1].splitlines()[0].strip()
    if not tail or len(tail.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        _fail("evidence report is empty or exceeds its size limit")
    report = _dict(strict_json(tail, "evidence report"), "evidence report")
    required = {"version", "repository", "run_id", "run_attempt", "commit", "tree", "suites"}
    if set(report) != required:
        _fail("evidence report fields do not match version 1")
    if type(report["version"]) is not int or report["version"] != 1:
        _fail("evidence report version is unsupported")
    suites = _list(report["suites"], "evidence suites")
    if suites != list(SUITES):
        _fail("evidence report suite inventory does not match full coverage")
    return report


def _runs_path(repository: str, commit: str) -> str:
    encoded_repo = quote(repository, safe="/")
    return (
        f"/repos/{encoded_repo}/actions/workflows/ci.yml/runs"
        f"?head_sha={commit}&per_page=100"
    )


def _commit_tree(api: Api, repository: str, commit: str) -> str:
    encoded_repo = quote(repository, safe="/")
    commit_path = f"/repos/{encoded_repo}/commits/{commit}"
    commit_body = _dict(_api_json(api, commit_path, "commit"), "commit response")
    if _sha(commit_body.get("sha"), "API commit") != commit:
        _fail("GitHub commit response does not match the requested commit")
    commit_data = _dict(commit_body.get("commit"), "commit data")
    api_tree = _dict(commit_data.get("tree"), "commit tree").get("sha")
    return _sha(api_tree, "API commit tree")


def _verify_completed_run(
    candidate: dict[str, object],
    repository: str,
    tree: str,
    api: Api,
) -> Selection:
    run_id = _positive_int(candidate.get("id"), "workflow run id")
    status = candidate.get("status")
    if status in {"requested", "queued", "pending", "waiting", "in_progress"}:
        raise PendingRunError(f"selected CI run {run_id} is {status!r}")
    if status != "completed":
        _fail("selected CI run has an invalid status")
    if candidate.get("conclusion") != "success":
        return Selection(False, "selected CI run was not successful", run_id)

    attempt = _positive_int(candidate.get("run_attempt"), "workflow run attempt")
    encoded_repo = quote(repository, safe="/")
    jobs_path = (
        f"/repos/{encoded_repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        "?per_page=100"
    )
    _, evidence_job_id = _validated_jobs(
        _api_json(api, jobs_path, "exact-attempt jobs"), attempt
    )
    logs_path = f"/repos/{encoded_repo}/actions/jobs/{evidence_job_id}/logs"
    report = _parse_report(_api_text(api, logs_path, "evidence job log"))

    if _repository(report["repository"]) != repository:
        _fail("evidence repository does not match")
    if _positive_int(report["run_id"], "evidence run_id") != run_id:
        _fail("evidence run_id does not match")
    if _positive_int(report["run_attempt"], "evidence run_attempt") != attempt:
        _fail("evidence run_attempt does not match")
    actual_commit = _sha(report["commit"], "evidence commit")
    if _sha(report["tree"], "evidence tree") != tree:
        _fail("evidence tree does not match the checked-out tree")

    if _commit_tree(api, repository, actual_commit) != tree:
        _fail("candidate commit tree does not match the tested merge tree")
    return Selection(True, "reused completed exact-tree CI evidence", run_id)


def _manual_selection(
    repository: str,
    commit: str,
    tree: str,
    api: Api,
) -> Selection:
    candidate = _latest_run(
        _api_json(api, _runs_path(repository, commit), "workflow runs"),
        repository,
        commit,
    )
    if candidate is None:
        return Selection(False, "no CI run exists for the exact commit")
    run_id = _positive_int(candidate.get("id"), "workflow run id")
    status = candidate.get("status")
    if status in {"requested", "queued", "pending", "waiting", "in_progress"}:
        raise PendingRunError(
            f"latest exact-commit CI run {run_id} is {status!r}; wait for it instead of starting duplicate full coverage"
        )
    if status != "completed":
        _fail("latest exact-commit CI run has an invalid status")
    if candidate.get("conclusion") != "success":
        return Selection(False, "latest exact-commit CI run was not successful", run_id)
    try:
        return _verify_completed_run(candidate, repository, tree, api)
    except EvidenceError as error:
        return Selection(False, str(error), run_id)


def _merged_pull_request(
    payload: object,
    repository: str,
    commit: str,
) -> dict[str, object] | None:
    pulls = _list(payload, "associated pull requests")
    if len(pulls) >= 100:
        _fail("associated pull-request response is truncated or ambiguous")
    matching: list[dict[str, object]] = []
    seen_numbers: set[int] = set()
    for raw in pulls:
        pull = _dict(raw, "associated pull request")
        number = _positive_int(pull.get("number"), "pull request number")
        if number in seen_numbers:
            _fail("associated pull-request response contains a duplicate")
        seen_numbers.add(number)
        base = _dict(pull.get("base"), "pull request base")
        head = _dict(pull.get("head"), "pull request head")
        base_repo = _dict(base.get("repo"), "pull request base repository")
        head_repo = _dict(head.get("repo"), "pull request head repository")
        head_sha = _sha(head.get("sha"), "pull request head commit")
        merged_at = pull.get("merged_at")
        if merged_at is not None:
            if not isinstance(merged_at, str) or not TIMESTAMP.fullmatch(merged_at):
                _fail("pull request merged_at is invalid")
            try:
                datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise EvidenceError("pull request merged_at is invalid") from error
        if (
            pull.get("state") == "closed"
            and merged_at is not None
            and pull.get("merge_commit_sha") == commit
            and base.get("ref") == "main"
            and base_repo.get("full_name") == repository
            and head_repo.get("full_name") == repository
        ):
            matching.append({**pull, "head_sha": head_sha})
    if not matching:
        return None
    if len(matching) != 1:
        _fail("multiple associated merged pull requests match the main commit")
    return matching[0]


def _documentation_base_selection(
    repository: str,
    base_commit: str,
    base_tree: str,
    api: Api,
) -> Selection:
    candidate = _latest_run(
        _api_json(api, _runs_path(repository, base_commit), "workflow runs"),
        repository,
        base_commit,
    )
    if candidate is None:
        return Selection(False, "no CI run exists for the documentation base")
    run_id = _positive_int(candidate.get("id"), "workflow run id")
    status = candidate.get("status")
    if status in {"requested", "queued", "pending", "waiting", "in_progress"}:
        return Selection(False, "documentation-base CI is still active", run_id)
    if status != "completed":
        return Selection(False, "documentation-base CI status is invalid", run_id)
    if candidate.get("conclusion") != "success":
        return Selection(False, "documentation-base CI was not successful", run_id)

    try:
        direct = _verify_completed_run(candidate, repository, base_tree, api)
    except EvidenceError as direct_error:
        try:
            explicitly_skipped = _has_explicit_full_matrix_skips(
                candidate, repository, api
            )
        except EvidenceError:
            return Selection(False, str(direct_error), run_id)
        if not explicitly_skipped:
            return Selection(False, str(direct_error), run_id)
        original = select_push_reuse(
            "push",
            "refs/heads/main",
            repository,
            base_commit,
            base_commit,
            base_tree,
            api,
        )
        if not original.reused:
            return Selection(
                False,
                f"documentation base original PR lacks full CI evidence: {original.reason}",
            )
        return Selection(
            False,
            "verified original full PR evidence for the documentation-only push base",
            original.run_id,
            True,
        )
    if not direct.reused:
        return Selection(False, direct.reason, run_id)
    return Selection(
        False,
        "verified complete CI evidence for the documentation-only push base",
        direct.run_id,
        True,
    )


def select_push_reuse(
    event: str,
    ref: str,
    repository: str,
    github_sha: str,
    commit: str,
    tree: str,
    api: Api = gh_api,
    *,
    prose_only: str = "false",
    push_before: str = "",
) -> Selection:
    """Select complete PR evidence for an exact same-repository main merge."""
    if event != "push":
        return Selection(False, "only push events may use main-push evidence reuse")
    if ref != "refs/heads/main":
        return Selection(False, "only pushes to refs/heads/main may reuse PR evidence")
    try:
        repository = _repository(repository)
        github_sha = _sha(github_sha, "GITHUB_SHA")
        commit = _sha(commit, "HEAD")
        tree = _sha(tree, "HEAD tree")
        if commit != github_sha:
            _fail("checked-out HEAD does not match GITHUB_SHA")
        if prose_only == "true":
            base_commit = _sha(push_before, "CI_PUSH_BEFORE")
            if base_commit == commit:
                _fail("documentation-only push base must differ from HEAD")
            base_tree = _commit_tree(api, repository, base_commit)
            return _documentation_base_selection(
                repository, base_commit, base_tree, api
            )
        encoded_repo = quote(repository, safe="/")
        pulls_path = (
            f"/repos/{encoded_repo}/commits/{commit}/pulls?per_page=100&page=1"
        )
        pull = _merged_pull_request(
            _api_json(api, pulls_path, "associated pull requests"),
            repository,
            commit,
        )
        if pull is None:
            return Selection(False, "no unique merged same-repository PR matches the main commit")
        head_sha = _sha(pull.get("head_sha"), "pull request head commit")
        candidate = _latest_run(
            _api_json(api, _runs_path(repository, head_sha), "workflow runs"),
            repository,
            head_sha,
            "pull_request",
        )
        if candidate is None:
            return Selection(False, "no pull-request CI run exists for the exact PR head")
        return _verify_completed_run(candidate, repository, tree, api)
    except EvidenceError as error:
        return Selection(False, str(error))


def select_reuse(
    event: str,
    repository: str,
    commit: str,
    tree: str,
    api: Api = gh_api,
) -> Selection:
    if event != "workflow_dispatch":
        return Selection(False, "only manual acceptance runs may reuse CI evidence")
    repository = _repository(repository)
    commit = _sha(commit, "candidate commit")
    tree = _sha(tree, "candidate tree")
    try:
        return _manual_selection(repository, commit, tree, api)
    except PendingRunError:
        raise
    except EvidenceError as error:
        return Selection(False, str(error))


def _append(path: str | None, text: str) -> None:
    if not path:
        _fail("required GitHub output path is unavailable")
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(text)


def _one_line(value: str) -> str:
    printable = "".join(
        character if 0x20 <= ord(character) <= 0x7E else " "
        for character in value
    )
    return " ".join(printable.split())


def select_cli(env: dict[str, str] | None = None, api: Api = gh_api) -> Selection:
    values = os.environ if env is None else env
    event = values.get("CI_EVENT", "")
    repository = _repository(values.get("GITHUB_REPOSITORY"))
    commit = _sha(_local_revision("HEAD"), "HEAD")
    tree = _sha(_local_revision("HEAD^{tree}"), "HEAD tree")
    try:
        result = select_reuse(event, repository, commit, tree, api)
    except PendingRunError as error:
        server = values.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        match = re.search(r"run (\d+)", str(error))
        if match:
            link = f"{server}/{repository}/actions/runs/{match.group(1)}"
            if values.get("GITHUB_STEP_SUMMARY"):
                _append(values["GITHUB_STEP_SUMMARY"], f"Exact-tree CI is still running: {link}\n")
        raise
    _append(values.get("GITHUB_OUTPUT"), f"reused={'true' if result.reused else 'false'}\n")
    summary = result.reason
    if result.run_id is not None:
        server = values.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        link = f"{server}/{repository}/actions/runs/{result.run_id}"
        summary += f": {link}"
    _append(values.get("GITHUB_STEP_SUMMARY"), summary + "\n")
    print(summary)
    return result


def select_push_cli(env: dict[str, str] | None = None, api: Api = gh_api) -> Selection:
    values = os.environ if env is None else env
    repository = values.get("GITHUB_REPOSITORY", "")
    try:
        commit = _local_revision("HEAD").strip()
        tree = _local_revision("HEAD^{tree}").strip()
        result = select_push_reuse(
            values.get("GITHUB_EVENT_NAME", ""),
            values.get("GITHUB_REF", ""),
            repository,
            values.get("GITHUB_SHA", ""),
            commit,
            tree,
            api,
            prose_only=values.get("CI_PROSE_ONLY", "false"),
            push_before=values.get("CI_PUSH_BEFORE", ""),
        )
    except EvidenceError as error:
        result = Selection(False, str(error))
    server = _one_line(
        values.get("GITHUB_SERVER_URL", "https://github.com")
    ).rstrip("/")
    source_run_id = (
        str(result.run_id)
        if (result.reused or result.docs_verified) and result.run_id
        else ""
    )
    source_run_url = (
        f"{server}/{repository}/actions/runs/{source_run_id}"
        if source_run_id
        else ""
    )
    reason = _one_line(result.reason)
    output = values.get("GITHUB_OUTPUT")
    _append(output, f"reused={'true' if result.reused else 'false'}\n")
    _append(output, f"docs_verified={'true' if result.docs_verified else 'false'}\n")
    _append(output, f"source_run_id={source_run_id}\n")
    _append(output, f"source_run_url={source_run_url}\n")
    _append(output, f"reason={reason}\n")
    summary = reason
    if source_run_url:
        summary += f": {source_run_url}"
    _append(values.get("GITHUB_STEP_SUMMARY"), summary + "\n")
    print(summary)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["emit"]:
        emit()
        return 0
    if arguments == ["select"]:
        select_cli()
        return 0
    if arguments == ["select-push"]:
        select_push_cli()
        return 0
    print(f"usage: {Path(sys.argv[0]).name} emit|select|select-push", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
