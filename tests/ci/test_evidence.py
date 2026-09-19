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
PR_HEAD = "c" * 40
TESTED_COMMIT = "d" * 40
BASE_COMMIT = "e" * 40
BASE_TREE = "f" * 40
BASE_TESTED_COMMIT = "1" * 40
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
    return runs_path_for(COMMIT)


def runs_path_for(commit: str) -> str:
    return (
        f"/repos/{REPOSITORY}/actions/workflows/ci.yml/runs"
        f"?head_sha={commit}&per_page=100"
    )


def pulls_path() -> str:
    return pulls_path_for(COMMIT)


def pulls_path_for(commit: str) -> str:
    return f"/repos/{REPOSITORY}/commits/{commit}/pulls?per_page=100&page=1"


def jobs_path() -> str:
    return jobs_path_for(RUN_ID, ATTEMPT)


def jobs_path_for(run_id: int, attempt: int) -> str:
    return (
        f"/repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}/jobs"
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
    event: str = "pull_request",
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
        "event": event,
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


def pull_request(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "number": 77,
        "state": "closed",
        "merged_at": "2026-09-18T16:00:00Z",
        "merge_commit_sha": COMMIT,
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        "head": {"sha": PR_HEAD, "repo": {"full_name": REPOSITORY}},
    }
    value.update(changes)
    return value


def successful_push_responses() -> dict[str, object]:
    all_jobs = jobs()
    return {
        pulls_path(): [pull_request()],
        runs_path_for(PR_HEAD): {
            "total_count": 1,
            "workflow_runs": [run(commit=PR_HEAD)],
        },
        jobs_path(): {"total_count": len(all_jobs), "jobs": all_jobs},
        logs_path(): report_log(report(commit=TESTED_COMMIT)),
        commit_path_for(TESTED_COMMIT): {
            "sha": TESTED_COMMIT,
            "commit": {"tree": {"sha": TREE}},
        },
    }


def successful_docs_push_responses() -> dict[str, object]:
    all_jobs = jobs()
    return {
        commit_path_for(BASE_COMMIT): {
            "sha": BASE_COMMIT,
            "commit": {"tree": {"sha": BASE_TREE}},
        },
        runs_path_for(BASE_COMMIT): {
            "total_count": 1,
            "workflow_runs": [run(commit=BASE_COMMIT)],
        },
        jobs_path(): {"total_count": len(all_jobs), "jobs": all_jobs},
        logs_path(): report_log(
            report(commit=BASE_TESTED_COMMIT, tree=BASE_TREE)
        ),
        commit_path_for(BASE_TESTED_COMMIT): {
            "sha": BASE_TESTED_COMMIT,
            "commit": {"tree": {"sha": BASE_TREE}},
        },
    }


class EvidenceTests(unittest.TestCase):
    def tearDown(self) -> None:
        evidence._gh_allows_escape_sequences.cache_clear()

    def test_gh_api_opts_in_only_for_job_logs_when_supported(self) -> None:
        log_path = logs_path()
        with mock.patch.object(
            evidence, "run_bounded",
            side_effect=["usage: gh api [--allow-escape-sequences]", "log"],
        ) as bounded:
            self.assertEqual(evidence.gh_api(log_path), "log")
        self.assertEqual(
            bounded.call_args_list,
            [
                mock.call(["gh", "api", "--help"]),
                mock.call(["gh", "api", "--allow-escape-sequences", log_path]),
            ],
        )

    def test_gh_api_legacy_cli_reads_job_logs_without_unknown_flag(self) -> None:
        log_path = logs_path()
        with mock.patch.object(
            evidence, "run_bounded", side_effect=["usage: gh api", "log"]
        ) as bounded:
            self.assertEqual(evidence.gh_api(log_path), "log")
        self.assertEqual(
            bounded.call_args_list,
            [mock.call(["gh", "api", "--help"]), mock.call(["gh", "api", log_path])],
        )

    def test_gh_api_does_not_probe_or_opt_in_for_json_requests(self) -> None:
        path = runs_path()
        with mock.patch.object(evidence, "run_bounded", return_value="{}") as bounded:
            self.assertEqual(evidence.gh_api(path), "{}")
        bounded.assert_called_once_with(["gh", "api", path])

    def test_gh_api_rejects_untrusted_and_malformed_log_paths(self) -> None:
        for path in (
            "https://api.github.com/repos/openai/project",
            "/repos/openai/project/../secret",
            "/repos/openai/project/actions/jobs/0/logs",
            "/repos/openai/project/actions/jobs/not-a-number/logs",
        ):
            with self.subTest(path=path), self.assertRaises(evidence.EvidenceError):
                evidence.gh_api(path)

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

    def test_suite_inventory_includes_optimizer_after_native_numerics(self) -> None:
        self.assertEqual(len(evidence.SUITES), 18)
        index = evidence.SUITES.index("native-numerics")
        self.assertEqual(evidence.SUITES[index + 1], "native-optimizer")

    def test_push_reuses_exact_pr_run_and_actual_tested_tree(self) -> None:
        api = FakeApi(successful_push_responses())
        result = evidence.select_push_reuse(
            "push", "refs/heads/main", REPOSITORY, COMMIT, COMMIT, TREE, api
        )
        self.assertTrue(result.reused)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(
            api.calls,
            [
                pulls_path(),
                runs_path_for(PR_HEAD),
                jobs_path(),
                logs_path(),
                commit_path_for(TESTED_COMMIT),
            ],
        )

    def test_docs_push_verifies_exact_base_full_run_without_code_reuse(self) -> None:
        api = FakeApi(successful_docs_push_responses())
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            api,
            prose_only="true",
            push_before=BASE_COMMIT,
        )
        self.assertFalse(result.reused)
        self.assertTrue(result.docs_verified)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(
            api.calls,
            [
                commit_path_for(BASE_COMMIT),
                runs_path_for(BASE_COMMIT),
                jobs_path(),
                logs_path(),
                commit_path_for(BASE_TESTED_COMMIT),
            ],
        )

    def test_docs_mode_requires_exact_lowercase_true(self) -> None:
        for value in ("TRUE", "1", "yes", "false", ""):
            with self.subTest(value=value):
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(successful_push_responses()),
                    prose_only=value,
                    push_before=BASE_COMMIT,
                )
                self.assertTrue(result.reused)
                self.assertFalse(result.docs_verified)

    def test_docs_pending_failed_or_missing_base_never_falls_back(self) -> None:
        cases: dict[str, object] = {
            "missing": {"total_count": 0, "workflow_runs": []},
            "pending": {
                "total_count": 1,
                "workflow_runs": [
                    run(commit=BASE_COMMIT, status="in_progress", conclusion=None)
                ],
            },
            "failed": {
                "total_count": 1,
                "workflow_runs": [run(commit=BASE_COMMIT, conclusion="failure")],
            },
        }
        for name, base_runs in cases.items():
            with self.subTest(case=name):
                api = FakeApi(
                    {
                        commit_path_for(BASE_COMMIT): {
                            "sha": BASE_COMMIT,
                            "commit": {"tree": {"sha": BASE_TREE}},
                        },
                        runs_path_for(BASE_COMMIT): base_runs,
                    }
                )
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    api,
                    prose_only="true",
                    push_before=BASE_COMMIT,
                )
                self.assertFalse(result.reused)
                self.assertFalse(result.docs_verified)
                self.assertNotIn(pulls_path_for(BASE_COMMIT), api.calls)

    def test_docs_malformed_or_incomplete_base_never_falls_back(self) -> None:
        malformed_commit = successful_docs_push_responses()
        malformed_commit[commit_path_for(BASE_COMMIT)] = {
            "sha": "2" * 40,
            "commit": {"tree": {"sha": BASE_TREE}},
        }
        incomplete_jobs = successful_docs_push_responses()
        altered = jobs()
        del altered[0]
        incomplete_jobs[jobs_path()] = {
            "total_count": len(altered),
            "jobs": altered,
        }
        for name, responses in (
            ("malformed-commit", malformed_commit),
            ("incomplete-jobs", incomplete_jobs),
        ):
            with self.subTest(case=name):
                api = FakeApi(responses)
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    api,
                    prose_only="true",
                    push_before=BASE_COMMIT,
                )
                self.assertFalse(result.reused)
                self.assertFalse(result.docs_verified)
                self.assertNotIn(pulls_path_for(BASE_COMMIT), api.calls)

    def test_docs_expanded_partial_skip_cannot_fallback_to_original_pr(self) -> None:
        base_run_id = RUN_ID + 10
        base_attempt = 1
        skipped_jobs = [
            {**job, "run_attempt": base_attempt, "conclusion": "skipped"}
            if (
                "Supported suite / " in str(job["name"])
                or str(job["name"]).endswith("Suite evidence")
            )
            else {**job, "run_attempt": base_attempt}
            for job in jobs()
        ]
        responses = successful_push_responses()
        responses.update(
            {
                commit_path_for(BASE_COMMIT): {
                    "sha": BASE_COMMIT,
                    "commit": {"tree": {"sha": BASE_TREE}},
                },
                runs_path_for(BASE_COMMIT): {
                    "total_count": 1,
                    "workflow_runs": [
                        run(
                            run_id=base_run_id,
                            attempt=base_attempt,
                            commit=BASE_COMMIT,
                            event="push",
                        )
                    ],
                },
                jobs_path_for(base_run_id, base_attempt): {
                    "total_count": len(skipped_jobs),
                    "jobs": skipped_jobs,
                },
                pulls_path_for(BASE_COMMIT): [
                    pull_request(merge_commit_sha=BASE_COMMIT)
                ],
                runs_path_for(PR_HEAD): {
                    "total_count": 1,
                    "workflow_runs": [run(commit=PR_HEAD)],
                },
                logs_path(): report_log(
                    report(commit=BASE_TESTED_COMMIT, tree=BASE_TREE)
                ),
                commit_path_for(BASE_TESTED_COMMIT): {
                    "sha": BASE_TESTED_COMMIT,
                    "commit": {"tree": {"sha": BASE_TREE}},
                },
            }
        )
        api = FakeApi(responses)
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            api,
            prose_only="true",
            push_before=BASE_COMMIT,
        )
        self.assertFalse(result.reused)
        self.assertFalse(result.docs_verified)
        self.assertNotIn(pulls_path_for(BASE_COMMIT), api.calls)

    def test_docs_actual_three_job_wrapper_skip_verifies_original_pr(self) -> None:
        base_run_id = RUN_ID + 10
        base_attempt = 1
        collapsed = [
            {
                **successful_job(9201, "CI topology"),
                "run_attempt": base_attempt,
            },
            {
                **successful_job(9202, "Full suites"),
                "run_attempt": base_attempt,
                "conclusion": "skipped",
            },
            {
                **successful_job(9203, evidence.FINAL_AGGREGATION),
                "run_attempt": base_attempt,
            },
        ]
        responses = successful_push_responses()
        responses.update(
            {
                commit_path_for(BASE_COMMIT): {
                    "sha": BASE_COMMIT,
                    "commit": {"tree": {"sha": BASE_TREE}},
                },
                runs_path_for(BASE_COMMIT): {
                    "total_count": 1,
                    "workflow_runs": [
                        run(
                            run_id=base_run_id,
                            attempt=base_attempt,
                            commit=BASE_COMMIT,
                            event="push",
                        )
                    ],
                },
                jobs_path_for(base_run_id, base_attempt): {
                    "total_count": 3,
                    "jobs": collapsed,
                },
                pulls_path_for(BASE_COMMIT): [
                    pull_request(merge_commit_sha=BASE_COMMIT)
                ],
                logs_path(): report_log(
                    report(commit=BASE_TESTED_COMMIT, tree=BASE_TREE)
                ),
                commit_path_for(BASE_TESTED_COMMIT): {
                    "sha": BASE_TESTED_COMMIT,
                    "commit": {"tree": {"sha": BASE_TREE}},
                },
            }
        )
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            FakeApi(responses),
            prose_only="true",
            push_before=BASE_COMMIT,
        )
        self.assertFalse(result.reused)
        self.assertTrue(result.docs_verified)
        self.assertEqual(result.run_id, RUN_ID)

    def test_docs_three_job_wrapper_skip_shape_is_exact(self) -> None:
        base_run_id = RUN_ID + 10
        base_attempt = 1
        base_jobs = [
            {**successful_job(9201, "CI topology"), "run_attempt": base_attempt},
            {
                **successful_job(9202, "Full suites"),
                "run_attempt": base_attempt,
                "conclusion": "skipped",
            },
            {
                **successful_job(9203, evidence.FINAL_AGGREGATION),
                "run_attempt": base_attempt,
            },
        ]
        cases = {
            "extra": base_jobs + [
                {**successful_job(9204, "unexpected"), "run_attempt": base_attempt}
            ],
            "blocking-success": [
                {**job, "conclusion": "success"}
                if job["name"] == "Full suites"
                else job
                for job in base_jobs
            ],
            "final-skipped": [
                {**job, "conclusion": "skipped"}
                if job["name"] == evidence.FINAL_AGGREGATION
                else job
                for job in base_jobs
            ],
            "wrong-attempt": [
                {**base_jobs[0], "run_attempt": 2}, *base_jobs[1:]
            ],
            "duplicate-id": [
                base_jobs[0], base_jobs[1], {**base_jobs[2], "id": base_jobs[0]["id"]}
            ],
        }
        for name, altered in cases.items():
            with self.subTest(case=name):
                responses = {
                    commit_path_for(BASE_COMMIT): {
                        "sha": BASE_COMMIT,
                        "commit": {"tree": {"sha": BASE_TREE}},
                    },
                    runs_path_for(BASE_COMMIT): {
                        "total_count": 1,
                        "workflow_runs": [
                            run(
                                run_id=base_run_id,
                                attempt=base_attempt,
                                commit=BASE_COMMIT,
                                event="push",
                            )
                        ],
                    },
                    jobs_path_for(base_run_id, base_attempt): {
                        "total_count": len(altered),
                        "jobs": altered,
                    },
                }
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                    prose_only="true",
                    push_before=BASE_COMMIT,
                )
                self.assertFalse(result.reused)
                self.assertFalse(result.docs_verified)

    def test_docs_skipped_base_rejects_skipped_original_pr_run(self) -> None:
        base_run_id = RUN_ID + 10
        base_attempt = 1
        skipped_jobs = [
            {**successful_job(9201, "CI topology"), "run_attempt": base_attempt},
            {
                **successful_job(9202, "Full suites"),
                "run_attempt": base_attempt,
                "conclusion": "skipped",
            },
            {
                **successful_job(9203, evidence.FINAL_AGGREGATION),
                "run_attempt": base_attempt,
            },
        ]
        responses = {
            commit_path_for(BASE_COMMIT): {
                "sha": BASE_COMMIT,
                "commit": {"tree": {"sha": BASE_TREE}},
            },
            runs_path_for(BASE_COMMIT): {
                "total_count": 1,
                "workflow_runs": [
                    run(
                        run_id=base_run_id,
                        attempt=base_attempt,
                        commit=BASE_COMMIT,
                        event="push",
                    )
                ],
            },
            jobs_path_for(base_run_id, base_attempt): {
                "total_count": len(skipped_jobs),
                "jobs": skipped_jobs,
            },
            pulls_path_for(BASE_COMMIT): [pull_request(merge_commit_sha=BASE_COMMIT)],
            runs_path_for(PR_HEAD): {
                "total_count": 1,
                "workflow_runs": [run(commit=PR_HEAD)],
            },
            jobs_path(): {"total_count": len(jobs()), "jobs": [
                {**job, "conclusion": "skipped"}
                if "Supported suite / " in str(job["name"])
                else job
                for job in jobs()
            ]},
        }
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            FakeApi(responses),
            prose_only="true",
            push_before=BASE_COMMIT,
        )
        self.assertFalse(result.reused)
        self.assertFalse(result.docs_verified)

    def test_push_requires_main_push_and_exact_checked_out_sha(self) -> None:
        cases = (
            ("workflow_dispatch", "refs/heads/main", COMMIT, COMMIT),
            ("push", "refs/heads/topic", COMMIT, COMMIT),
            ("push", "refs/heads/main", COMMIT, "e" * 40),
        )
        for event, ref, github_sha, local_head in cases:
            with self.subTest(event=event, ref=ref, local_head=local_head):
                api = FakeApi({})
                result = evidence.select_push_reuse(
                    event,
                    ref,
                    REPOSITORY,
                    github_sha,
                    local_head,
                    TREE,
                    api,
                )
                self.assertFalse(result.reused)
                self.assertEqual(api.calls, [])

    def test_push_requires_unique_merged_same_repo_main_pr(self) -> None:
        cases: dict[str, object] = {
            "none": [],
            "unmerged": [pull_request(merged_at=None)],
            "open": [pull_request(state="open")],
            "wrong-merge": [pull_request(merge_commit_sha="e" * 40)],
            "wrong-base": [
                pull_request(
                    base={"ref": "release", "repo": {"full_name": REPOSITORY}}
                )
            ],
            "foreign-head": [
                pull_request(
                    head={"sha": PR_HEAD, "repo": {"full_name": "fork/project"}}
                )
            ],
            "foreign-base": [
                pull_request(
                    base={"ref": "main", "repo": {"full_name": "fork/project"}}
                )
            ],
            "ambiguous": [pull_request(), pull_request(number=78)],
            "duplicate": [pull_request(), pull_request()],
            "malformed": [{"number": 77}],
            "truncated": [pull_request(number=index + 1) for index in range(100)],
        }
        for name, pulls in cases.items():
            with self.subTest(case=name):
                api = FakeApi({pulls_path(): pulls})
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    api,
                )
                self.assertFalse(result.reused)
                self.assertEqual(api.calls, [pulls_path()])

    def test_push_requires_pull_request_event_for_exact_pr_head(self) -> None:
        for event in ("push", "workflow_dispatch", "schedule"):
            with self.subTest(event=event):
                responses = successful_push_responses()
                responses[runs_path_for(PR_HEAD)] = {
                    "total_count": 1,
                    "workflow_runs": [run(commit=PR_HEAD, event=event)],
                }
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_push_selects_latest_pull_request_run_not_other_supported_event(self) -> None:
        other = run(
            run_id=RUN_ID + 1,
            attempt=1,
            created_at="2026-09-18T16:00:00Z",
            commit=PR_HEAD,
            event="push",
        )
        responses = successful_push_responses()
        responses[runs_path_for(PR_HEAD)] = {
            "total_count": 2,
            "workflow_runs": [run(commit=PR_HEAD), other],
        }
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            FakeApi(responses),
        )
        self.assertTrue(result.reused)

    def test_push_latest_failed_pr_run_blocks_older_green_pr_run(self) -> None:
        older = run(
            run_id=RUN_ID - 1,
            attempt=1,
            created_at="2026-09-18T14:00:00Z",
            commit=PR_HEAD,
        )
        latest = run(commit=PR_HEAD, conclusion="failure")
        responses = successful_push_responses()
        responses[runs_path_for(PR_HEAD)] = {
            "total_count": 2,
            "workflow_runs": [older, latest],
        }
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            FakeApi(responses),
        )
        self.assertFalse(result.reused)
        self.assertEqual(result.run_id, RUN_ID)

    def test_push_pending_failed_or_ambiguous_source_falls_back_fresh(self) -> None:
        cases = (
            run(commit=PR_HEAD, status="in_progress", conclusion=None),
            run(commit=PR_HEAD, conclusion="failure"),
            run(commit=PR_HEAD, status="mystery", conclusion=None),
        )
        for source_run in cases:
            with self.subTest(source_run=source_run):
                responses = successful_push_responses()
                responses[runs_path_for(PR_HEAD)] = {
                    "total_count": 1,
                    "workflow_runs": [source_run],
                }
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_push_rejects_skipped_or_chained_reuse_job_set(self) -> None:
        cases: dict[str, list[dict[str, object]]] = {}
        skipped = jobs()
        skipped[0] = {**skipped[0], "conclusion": "skipped"}
        cases["skipped-suite"] = skipped
        no_suites = [job for job in jobs() if "Supported suite / " not in str(job["name"])]
        cases["no-suite-jobs"] = no_suites
        missing_optimizer = [
            job
            for job in jobs()
            if not str(job["name"]).endswith("Supported suite / native-optimizer")
        ]
        cases["missing-optimizer"] = missing_optimizer
        for name, altered in cases.items():
            with self.subTest(case=name):
                responses = successful_push_responses()
                responses[jobs_path()] = {
                    "total_count": len(altered),
                    "jobs": altered,
                }
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

    def test_push_rejects_stale_report_or_api_tested_tree(self) -> None:
        for field, value in (
            ("tree", "e" * 40),
            ("suites", list(evidence.SUITES[:-1])),
            ("repository", "other/repository"),
        ):
            with self.subTest(field=field):
                responses = successful_push_responses()
                responses[logs_path()] = report_log(
                    report(commit=TESTED_COMMIT, **{field: value})
                )
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)
        responses = successful_push_responses()
        responses[commit_path_for(TESTED_COMMIT)] = {
            "sha": TESTED_COMMIT,
            "commit": {"tree": {"sha": "e" * 40}},
        }
        result = evidence.select_push_reuse(
            "push",
            "refs/heads/main",
            REPOSITORY,
            COMMIT,
            COMMIT,
            TREE,
            FakeApi(responses),
        )
        self.assertFalse(result.reused)

    def test_push_api_failure_or_truncated_run_list_falls_back_fresh(self) -> None:
        for response in (
            RuntimeError("offline"),
            {"total_count": 101, "workflow_runs": [run(commit=PR_HEAD)]},
        ):
            with self.subTest(response=type(response).__name__):
                responses = successful_push_responses()
                responses[runs_path_for(PR_HEAD)] = response
                result = evidence.select_push_reuse(
                    "push",
                    "refs/heads/main",
                    REPOSITORY,
                    COMMIT,
                    COMMIT,
                    TREE,
                    FakeApi(responses),
                )
                self.assertFalse(result.reused)

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

    def test_terminal_controls_around_evidence_are_parsed_but_never_rendered(self) -> None:
        responses = successful_responses()
        responses[logs_path()] = "\x1b[31mrunner preface\x1b[0m\n" + report_log() + "\x07"
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "CI_EVENT": "workflow_dispatch",
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
            }
            stdout = io.StringIO()
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ), redirect_stdout(stdout):
                result = evidence.select_cli(env, FakeApi(responses))
            summary = summary_path.read_text(encoding="utf-8")
        self.assertTrue(result.reused)
        self.assertNotIn("\x1b", stdout.getvalue())
        self.assertNotIn("\x07", stdout.getvalue())
        self.assertEqual(stdout.getvalue(), summary)

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
            stdout = io.StringIO()
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ), redirect_stdout(stdout):
                result = evidence.select_cli(env, api)
            self.assertTrue(result.reused)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "reused=true\n")
            expected = (
                "reused completed exact-tree CI evidence: "
                f"https://github.example/{REPOSITORY}/actions/runs/{RUN_ID}\n"
            )
            self.assertEqual(summary_path.read_text(encoding="utf-8"), expected)
            self.assertEqual(stdout.getvalue(), expected)

    def test_select_push_cli_writes_verified_source_and_reason_outputs(self) -> None:
        api = FakeApi(successful_push_responses())
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": COMMIT,
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
                "GITHUB_SERVER_URL": "https://github.example",
            }
            stdout = io.StringIO()
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ), redirect_stdout(stdout):
                result = evidence.select_push_cli(env, api)
            expected_url = (
                f"https://github.example/{REPOSITORY}/actions/runs/{RUN_ID}"
            )
            self.assertTrue(result.reused)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "reused=true\n"
                "docs_verified=false\n"
                f"source_run_id={RUN_ID}\n"
                f"source_run_url={expected_url}\n"
                "reason=reused completed exact-tree CI evidence\n",
            )
            expected_summary = (
                f"reused completed exact-tree CI evidence: {expected_url}\n"
            )
            self.assertEqual(summary_path.read_text(encoding="utf-8"), expected_summary)
            self.assertEqual(stdout.getvalue(), expected_summary)

    def test_select_push_cli_writes_distinct_docs_verification_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": COMMIT,
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
                "CI_PROSE_ONLY": "true",
                "CI_PUSH_BEFORE": BASE_COMMIT,
            }
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ):
                result = evidence.select_push_cli(
                    env, FakeApi(successful_docs_push_responses())
                )
            output = output_path.read_text(encoding="utf-8")
            self.assertFalse(result.reused)
            self.assertTrue(result.docs_verified)
            self.assertIn("reused=false\n", output)
            self.assertIn("docs_verified=true\n", output)
            self.assertIn(f"source_run_id={RUN_ID}\n", output)
            self.assertIn(
                f"source_run_url=https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}\n",
                output,
            )

    def test_select_push_cli_pending_source_outputs_fresh_without_source_claim(self) -> None:
        responses = successful_push_responses()
        responses[runs_path_for(PR_HEAD)] = {
            "total_count": 1,
            "workflow_runs": [
                run(commit=PR_HEAD, status="in_progress", conclusion=None)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": COMMIT,
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
            }
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ):
                result = evidence.select_push_cli(env, FakeApi(responses))
            output = output_path.read_text(encoding="utf-8")
            self.assertFalse(result.reused)
            self.assertIn("reused=false\n", output)
            self.assertIn("docs_verified=false\n", output)
            self.assertIn("source_run_id=\n", output)
            self.assertIn("source_run_url=\n", output)
            self.assertIn("reason=selected CI run", output)

    def test_select_push_cli_local_git_uncertainty_falls_back_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": COMMIT,
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
            }
            with mock.patch.object(
                evidence,
                "_local_revision",
                side_effect=evidence.EvidenceError("git unavailable"),
            ):
                result = evidence.select_push_cli(env, FakeApi({}))
            self.assertFalse(result.reused)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "reused=false\n"
                "docs_verified=false\n"
                "source_run_id=\n"
                "source_run_url=\n"
                "reason=git unavailable\n",
            )

    def test_select_push_cli_does_not_render_untrusted_job_controls(self) -> None:
        responses = successful_push_responses()
        altered = jobs()
        altered[0] = {
            **altered[0],
            "name": "\x1b[31mhostile\x07\nname",
            "conclusion": "failure",
        }
        responses[jobs_path()] = {"total_count": len(altered), "jobs": altered}
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "output"
            summary_path = Path(temporary) / "summary"
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": COMMIT,
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_STEP_SUMMARY": str(summary_path),
            }
            stdout = io.StringIO()
            with mock.patch.object(
                evidence, "_local_revision", side_effect=[COMMIT, TREE]
            ), redirect_stdout(stdout):
                result = evidence.select_push_cli(env, FakeApi(responses))
            rendered = (
                output_path.read_text(encoding="utf-8")
                + summary_path.read_text(encoding="utf-8")
                + stdout.getvalue()
            )
            self.assertFalse(result.reused)
            self.assertNotIn("\x1b", rendered)
            self.assertNotIn("\x07", rendered)
            self.assertNotIn("\nname", rendered)

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
