from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.ci import evidence


REPOSITORY = "openai/eshkol-transformer"
COMMIT = "a" * 40
TREE = "b" * 40
RUN_ID = 4815162342
ATTEMPT = 3
EVIDENCE_JOB_ID = 9001


class FakeApi:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, path: str) -> str | bytes:
        self.calls.append(path)
        if path not in self.responses:
            raise AssertionError(f"unexpected API path: {path}")
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, (str, bytes)):
            return response
        return json.dumps(response, separators=(",", ":"))


def runs_path() -> str:
    return (
        f"/repos/{REPOSITORY}/actions/workflows/ci.yml/runs"
        f"?head_sha={COMMIT}&per_page=100"
    )


def jobs_path() -> str:
    return (
        f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/{ATTEMPT}/jobs"
        "?per_page=100"
    )


def logs_path() -> str:
    return f"/repos/{REPOSITORY}/actions/jobs/{EVIDENCE_JOB_ID}/logs"


def commit_path() -> str:
    return commit_path_for(COMMIT)


def commit_path_for(commit: str) -> str:
    return f"/repos/{REPOSITORY}/commits/{commit}"


def run(
    *,
    run_id: int = RUN_ID,
    attempt: int = ATTEMPT,
    status: str = "completed",
    conclusion: str | None = "success",
    created_at: str = "2026-09-18T15:00:00Z",
    commit: str = COMMIT,
) -> dict[str, object]:
    return {
        "id": run_id,
        "run_attempt": attempt,
        "status": status,
        "conclusion": conclusion,
        "created_at": created_at,
        "head_sha": commit,
        "repository": {"full_name": REPOSITORY},
        "path": evidence.CI_WORKFLOW_PATH + "@refs/heads/main",
        "event": "pull_request",
    }


def successful_job(job_id: int, name: str) -> dict[str, object]:
    return {
        "id": job_id,
        "name": name,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": ATTEMPT,
    }


def jobs() -> list[dict[str, object]]:
    result = [
        successful_job(index + 1, f"Full suites / Supported suite / {suite}")
        for index, suite in enumerate(evidence.SUITES)
    ]
    result.extend(
        [
            successful_job(EVIDENCE_JOB_ID, "Full suites / Suite evidence"),
            successful_job(9002, evidence.FINAL_AGGREGATION),
            successful_job(9003, "CI topology"),
        ]
    )
    return result


def report(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "repository": REPOSITORY,
        "run_id": RUN_ID,
        "run_attempt": ATTEMPT,
        "commit": COMMIT,
        "tree": TREE,
        "suites": list(evidence.SUITES),
    }
    value.update(changes)
    return value


def report_log(value: dict[str, object] | None = None) -> str:
    payload = report() if value is None else value
    return (
        "2026-09-18T15:30:00Z runner preface\n"
        f"2026-09-18T15:30:01Z {evidence.PREFIX}"
        + json.dumps(payload, separators=(",", ":"))
        + "\n2026-09-18T15:30:02Z runner suffix\n"
    )


def successful_responses() -> dict[str, object]:
    all_jobs = jobs()
    return {
        runs_path(): {"total_count": 1, "workflow_runs": [run()]},
        jobs_path(): {"total_count": len(all_jobs), "jobs": all_jobs},
        logs_path(): report_log(),
        commit_path(): {"sha": COMMIT, "commit": {"tree": {"sha": TREE}}},
    }


class EvidenceTests(unittest.TestCase):
    def test_successfully_reuses_exact_run_attempt_tree_and_suite_set(self) -> None:
        api = FakeApi(successful_responses())
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertTrue(result.reused)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(
            api.calls, [runs_path(), jobs_path(), logs_path(), commit_path()]
        )

    def test_schedule_always_runs_fresh_without_network(self) -> None:
        api = FakeApi({})
        result = evidence.select_reuse("schedule", REPOSITORY, COMMIT, TREE, api)
        self.assertFalse(result.reused)
        self.assertEqual(api.calls, [])

    def test_missing_run_falls_back_fresh(self) -> None:
        api = FakeApi(
            {runs_path(): {"total_count": 0, "workflow_runs": []}}
        )
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertFalse(result.reused)
        self.assertEqual(api.calls, [runs_path()])

    def test_latest_failed_run_blocks_reuse_of_older_green(self) -> None:
        latest = run(run_id=RUN_ID, conclusion="failure")
        older = run(
            run_id=RUN_ID - 1,
            attempt=1,
            created_at="2026-09-18T14:00:00Z",
        )
        api = FakeApi(
            {runs_path(): {"total_count": 2, "workflow_runs": [older, latest]}}
        )
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertFalse(result.reused)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(api.calls, [runs_path()])

    def test_latest_run_uses_timestamp_value_not_api_order_or_text_order(self) -> None:
        latest = run(created_at="2026-09-18T15:00:00.900Z", conclusion="failure")
        older = run(
            run_id=RUN_ID - 1,
            attempt=1,
            created_at="2026-09-18T15:00:00Z",
        )
        api = FakeApi(
            {runs_path(): {"total_count": 2, "workflow_runs": [older, latest]}}
        )
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertFalse(result.reused)
        self.assertEqual(result.run_id, RUN_ID)

    def test_latest_pending_run_fails_instead_of_starting_duplicate(self) -> None:
        latest = run(status="in_progress", conclusion=None)
        older = run(
            run_id=RUN_ID - 1,
            attempt=1,
            created_at="2026-09-18T14:00:00Z",
        )
        api = FakeApi(
            {runs_path(): {"total_count": 2, "workflow_runs": [older, latest]}}
        )
        with self.assertRaisesRegex(evidence.PendingRunError, "wait for it"):
            evidence.select_reuse(
                "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
            )
        self.assertEqual(api.calls, [runs_path()])

    def test_stale_candidate_sha_is_not_reused(self) -> None:
        stale = run(commit="c" * 40)
        api = FakeApi(
            {runs_path(): {"total_count": 1, "workflow_runs": [stale]}}
        )
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertFalse(result.reused)

    def test_malformed_nonactive_run_status_falls_back_instead_of_waiting(self) -> None:
        malformed = run(status="mystery", conclusion=None)
        api = FakeApi(
            {runs_path(): {"total_count": 1, "workflow_runs": [malformed]}}
        )
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
        )
        self.assertFalse(result.reused)
        self.assertIn("invalid status", result.reason)

    def test_stale_report_fields_fall_back_fresh(self) -> None:
        for field, value in (
            ("commit", "c" * 40),
            ("tree", "d" * 40),
            ("run_id", RUN_ID + 1),
            ("run_attempt", ATTEMPT + 1),
            ("repository", "other/repository"),
            ("suites", list(evidence.SUITES[:-1])),
        ):
            with self.subTest(field=field):
                responses = successful_responses()
                responses[logs_path()] = report_log(report(**{field: value}))
                result = evidence.select_reuse(
                    "workflow_dispatch",
                    REPOSITORY,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_api_tree_must_match_tested_merge_tree(self) -> None:
        responses = successful_responses()
        responses[commit_path()] = {
            "sha": COMMIT,
            "commit": {"tree": {"sha": "d" * 40}},
        }
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertFalse(result.reused)

    def test_commit_api_must_return_the_reported_actual_commit(self) -> None:
        responses = successful_responses()
        responses[commit_path()] = {
            "sha": "c" * 40,
            "commit": {"tree": {"sha": TREE}},
        }
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertFalse(result.reused)

    def test_pr_head_can_reuse_different_tested_commit_with_same_tree(self) -> None:
        merge_commit = "c" * 40
        responses = successful_responses()
        responses[logs_path()] = report_log(report(commit=merge_commit))
        del responses[commit_path()]
        responses[commit_path_for(merge_commit)] = {
            "sha": merge_commit,
            "commit": {"tree": {"sha": TREE}},
        }
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertTrue(result.reused)

    def test_pr_merge_commit_with_different_tree_is_not_reused(self) -> None:
        merge_commit = "c" * 40
        responses = successful_responses()
        responses[logs_path()] = report_log(report(commit=merge_commit))
        del responses[commit_path()]
        responses[commit_path_for(merge_commit)] = {
            "sha": merge_commit,
            "commit": {"tree": {"sha": "d" * 40}},
        }
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertFalse(result.reused)

    def test_workflow_path_and_event_are_explicitly_allowlisted(self) -> None:
        for field, value in (
            ("path", ".github/workflows/other.yml"),
            ("event", "schedule"),
        ):
            with self.subTest(field=field):
                candidate = run()
                candidate[field] = value
                api = FakeApi(
                    {
                        runs_path(): {
                            "total_count": 1,
                            "workflow_runs": [candidate],
                        }
                    }
                )
                result = evidence.select_reuse(
                    "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
                )
                self.assertFalse(result.reused)

    def test_exact_attempt_job_mismatch_falls_back_fresh(self) -> None:
        responses = successful_responses()
        altered = jobs()
        altered[0]["run_attempt"] = ATTEMPT - 1
        responses[jobs_path()] = {"total_count": len(altered), "jobs": altered}
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertFalse(result.reused)

    def test_duplicate_or_malformed_evidence_falls_back_fresh(self) -> None:
        malformed_logs = (
            report_log() + report_log(),
            f'{evidence.PREFIX}{{"version":1,"version":1}}\n',
            f"{evidence.PREFIX}not-json\n",
            f"{evidence.PREFIX}{'x' * (evidence.MAX_EVIDENCE_BYTES + 1)}\n",
        )
        for logs in malformed_logs:
            with self.subTest(logs=logs[:80]):
                responses = successful_responses()
                responses[logs_path()] = logs
                result = evidence.select_reuse(
                    "workflow_dispatch",
                    REPOSITORY,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_incomplete_duplicate_or_unsuccessful_jobs_fall_back_fresh(self) -> None:
        cases: dict[str, list[dict[str, object]]] = {}
        missing = jobs()
        del missing[0]
        cases["missing"] = missing
        duplicate = jobs()
        duplicate.append(successful_job(9100, duplicate[0]["name"]))
        cases["duplicate-suite"] = duplicate
        unknown = jobs()
        unknown.append(successful_job(9101, "Supported suite / untrusted"))
        cases["unknown-suite"] = unknown
        skipped = jobs()
        skipped[0] = {**skipped[0], "conclusion": "skipped"}
        cases["skipped"] = skipped
        cancelled = jobs()
        cancelled[0] = {**cancelled[0], "conclusion": "cancelled"}
        cases["cancelled"] = cancelled
        pending = jobs()
        pending[0] = {**pending[0], "status": "in_progress", "conclusion": None}
        cases["pending"] = pending
        missing_evidence = jobs()
        del missing_evidence[-3]
        cases["missing-evidence"] = missing_evidence
        missing_aggregation = jobs()
        del missing_aggregation[-2]
        cases["missing-aggregation"] = missing_aggregation
        for name, altered in cases.items():
            with self.subTest(case=name):
                responses = successful_responses()
                responses[jobs_path()] = {
                    "total_count": len(altered),
                    "jobs": altered,
                }
                result = evidence.select_reuse(
                    "workflow_dispatch",
                    REPOSITORY,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_more_than_one_page_of_jobs_is_rejected(self) -> None:
        responses = successful_responses()
        responses[jobs_path()] = {"total_count": 101, "jobs": jobs()}
        result = evidence.select_reuse(
            "workflow_dispatch", REPOSITORY, COMMIT, TREE, FakeApi(responses)
        )
        self.assertFalse(result.reused)

    def test_incomplete_or_overlimit_workflow_run_page_is_rejected(self) -> None:
        for total in (2, 101):
            with self.subTest(total=total):
                api = FakeApi(
                    {
                        runs_path(): {
                            "total_count": total,
                            "workflow_runs": [run()],
                        }
                    }
                )
                result = evidence.select_reuse(
                    "workflow_dispatch", REPOSITORY, COMMIT, TREE, api
                )
                self.assertFalse(result.reused)
                self.assertEqual(api.calls, [runs_path()])

    def test_api_failure_and_oversized_response_fail_closed(self) -> None:
        for response in (
            RuntimeError("network unavailable"),
            "x" * (evidence.MAX_COMMAND_BYTES + 1),
        ):
            with self.subTest(response=type(response).__name__):
                result = evidence.select_reuse(
                    "workflow_dispatch",
                    REPOSITORY,
                    COMMIT,
                    TREE,
                    FakeApi({runs_path(): response}),
                )
                self.assertFalse(result.reused)

    def test_duplicate_key_in_api_json_falls_back_fresh(self) -> None:
        duplicate = '{"total_count":1,"total_count":1,"workflow_runs":[]}'
        result = evidence.select_reuse(
            "workflow_dispatch",
            REPOSITORY,
            COMMIT,
            TREE,
            FakeApi({runs_path(): duplicate}),
        )
        self.assertFalse(result.reused)

    def test_emit_uses_head_tree_and_strict_run_environment(self) -> None:
        env = {
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_RUN_ID": str(RUN_ID),
            "GITHUB_RUN_ATTEMPT": str(ATTEMPT),
            "GITHUB_SHA": COMMIT,
            "GITHUB_JOB": "evidence",
        }
        output = io.StringIO()
        with mock.patch.object(
            evidence, "_local_revision", side_effect=[COMMIT, TREE]
        ), redirect_stdout(output):
            evidence.emit(env)
        line = output.getvalue().strip()
        self.assertTrue(line.startswith(evidence.PREFIX))
        self.assertEqual(
            evidence.strict_json(line.removeprefix(evidence.PREFIX), "test report"),
            report(),
        )

    def test_emit_rejects_non_evidence_job_and_checkout_mismatch(self) -> None:
        base = {
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_RUN_ID": str(RUN_ID),
            "GITHUB_RUN_ATTEMPT": str(ATTEMPT),
            "GITHUB_SHA": COMMIT,
            "GITHUB_JOB": "evidence",
        }
        for changes in ({"GITHUB_JOB": "suites"}, {"GITHUB_SHA": "c" * 40}):
            with self.subTest(changes=changes), mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ):
                with self.assertRaises(evidence.EvidenceError):
                    evidence.make_report({**base, **changes})

    def test_select_cli_writes_output_and_actual_run_summary_link(self) -> None:
        api = FakeApi(successful_responses())
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "CI_EVENT": "workflow_dispatch",
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
                "GITHUB_SERVER_URL": "https://github.example",
            }
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ):
                result = evidence.select_cli(env, api)
            self.assertTrue(result.reused)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "reused=true\n")
            self.assertIn(
                f"https://github.example/{REPOSITORY}/actions/runs/{RUN_ID}",
                summary_path.read_text(encoding="utf-8"),
            )

    def test_pending_cli_does_not_claim_reuse(self) -> None:
        responses = successful_responses()
        responses[runs_path()] = {
            "total_count": 1,
            "workflow_runs": [run(status="queued", conclusion=None)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "CI_EVENT": "workflow_dispatch",
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
                "GITHUB_SERVER_URL": "https://github.com",
            }
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ):
                with self.assertRaises(evidence.PendingRunError):
                    evidence.select_cli(env, FakeApi(responses))
            self.assertFalse(output_path.exists())
            self.assertIn(str(RUN_ID), summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
