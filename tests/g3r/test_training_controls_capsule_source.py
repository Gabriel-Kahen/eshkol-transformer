"""Private control-provenance source gates; generated runtime remains unproved."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "native/tr3_lease_core_extension.esk"
STEP = ROOT / "native/tr3_step_transaction_extension.esk"
TAIL = ROOT / "native/tr3_step_commit_tail_extension.esk"
RESTORE = ROOT / "native/tr3_c_joint_restore_extension.esk"
TRAIN = ROOT / "native/tr3_public_train_extension.esk"
WRAPPER = ROOT / "native/g3r_training_controls_extension.esk"
AGGREGATE = ROOT / "native/g3r_training_controls_root.esk"
MANIFEST = ROOT / "native/g3r_training_controls_source_closure.txt"


def forms(path):
    return parse(path.read_text())


def define(path, name):
    matches = [f for f in forms(path) if isinstance(f, list) and f[:2] == ["define", name]]
    if len(matches) != 1:
        raise AssertionError(f"missing/duplicate {name}")
    return matches[0]


def source_slice(path, start, end):
    s = path if isinstance(path, str) else path.read_text()
    return s[s.index(start):s.index(end, s.index(start))]


def descendants(node, head):
    if isinstance(node, list):
        if node[:1] == [head]:
            yield node
        for child in node:
            yield from descendants(child, head)


def branch(route, name):
    for item in route[2][2][2][2][1:]:
        if item[0] == ["eq?", "mode", ["quote", name]]:
            return item
    raise AssertionError(f"route mode {name} absent")


def check_train_source(train):
    entry = source_slice(train, "(define (tr3-train-run-internal", "(define (trainer-train!")
    gate = entry.find("(tr3-controls-route 'guard record 'train")
    begin = entry.find("(tr3-metrics-train-begin-internal!)")
    if gate == -1 or begin == -1 or gate > begin:
        raise ValueError("direct trainer-train metrics mutate before capsule denial")


def check_source(core, step, tail, restore, wrapper, aggregate, manifest):
    f = parse(core)[-1]
    if f[:2] != ["define", "tr3-controls-route"] or f[2][0] != "let":
        raise ValueError("fixed core lexical issuer lost")
    if [x[0] for x in f[2][1]] != ["issued", "pending", "tail", "return-cell"]:
        raise ValueError("issuer or fixed-tail root is no longer lexical")
    if f[2][2][0] != "let*":
        raise ValueError("ordered private helpers changed")
    helper_names = [x[0] for x in f[2][2][1]]
    if helper_names != ["fail", "find-shell", "find-record", "record-absent?", "capacity",
                        "copy-bytes", "copy-controls", "copy-detached",
                        "same-controls?", "live-row", "live-originals?",
                        "clear-tail!"]:
        raise ValueError("issuer helper closure changed")
    route = f
    required = {"create", "guard", "stage-step", "stage-restore",
                "commit-tail", "commit-return", "check-current", "released", "publish",
                "retire"}
    for name in required:
        branch(route, name)
    enter_text = str(branch(route, "enter"))
    if "tr3-metrics" in enter_text:
        raise ValueError("capsule enter mutates metrics before journal admission")
    if not ("['same-controls?', ['vector-ref', 'row', '4'], 'current']" in enter_text
            and "['o2-optimizer-completed-updates-internal'" in enter_text
            and enter_text.find("['same-controls?'") <
            enter_text.find("['vector-set!', 'row', '6', 'b']")):
        raise ValueError("prior control journal not checked before capsule enter")
    grant = str(branch(route, "guard"))
    if not ("['vector-ref', 'row', '7']" in grant and
            "['vector-set!', 'row', '7', '#f']" in grant):
        raise ValueError("same-phase lower call grant is reusable")
    # A stale/copy shell is never admitted by its fields: only an issuer-list
    # exact identity search can find it. The source evaluator below exercises
    # this predicate with distinct but equal-shaped objects.
    lookup = str(f[2][2][1][1])
    if "['eq?', 'shell', ['vector-ref', ['car', 'rest'], '0']]" not in lookup:
        raise ValueError("issuer shell lookup lost identity")
    creation = branch(route, "create")
    creation_text = str(creation)
    stages = ["['vector-set!', 'pending', '0', 'row']",
              "['tr3-lease-create-internal', 'a', 'b', 'c', 'd', 'e']",
              "['vector-set!', 'row', '1', 'trainer']",
              "['tr3-lease-record-for-shell-internal'",
              "['set!', 'issued', 'next']"]
    positions = [creation_text.find(x) for x in stages]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("genuine create cleanup root moved after lookup")
    if "['tr3-lease-unenroll-internal!', 'trainer']" not in creation_text:
        raise ValueError("unpublished trainer cleanup disappeared")
    if "['tr3-lease-fail-stop', '134']" not in creation_text:
        raise ValueError("uncertain close no longer fails stop")
    if "['vector-copy', ['vector-ref', 'record', '11']]" not in creation_text:
        raise ValueError("14-handle original snapshot became mutable alias")
    if "['vector-copy', ['vector-ref', 'record', '10']]" not in creation_text:
        raise ValueError("module original snapshot became mutable alias")
    originals_text = str(branch(route, "originals"))
    for slot in ("7", "8", "10"):
        if f"['vector-copy', ['vector-ref', 'o', '{slot}']]" not in originals_text:
            raise ValueError("caller received mutable issuer roster alias")
    if "(record-absent? (vector-ref row 2))" not in core:
        raise ValueError("released capsule lacks genuine unenroll proof")
    for section, gate, first_write in [
        (source_slice(step, "(define (tr3-step-start-internal", "(define (tr3-step-prepare-internal!"),
         "'guard record 'step", "(vector-set! tr3-step-frames 0 successor)"),
        (source_slice(core, "(define (tr3-lease-restore-enter-internal!", "(define (tr3-lease-restore-recheck-internal"),
         "'guard record 'restore", "(vector-set! record 4 parent)"),
        (source_slice(core, "(define (tr3-lease-unenroll-internal!", "(define (tr3-lease-restore-enter-admission!"),
         "'guard record 'release", "(set-cdr! previous (cdr remaining))"),
    ]:
        if section.find(gate) == -1 or section.find(gate) > section.find(first_write):
            raise ValueError(f"enrolled mutator guard moved after write: {gate}")
    if "(tr3-controls-route 'guard record 'evaluate" not in core:
        raise ValueError("enrolled evaluation bypass introduced")
    check_train_source(TRAIN.read_text())
    step_pre = source_slice(tail, "(define (tr3-step-commit-with-epochs-and-precommit-internal!", "(define (tr3-step-commit-with-epochs-internal!")
    if not (step_pre.index("(precommit)") <
            step_pre.index("'stage-step") <
            step_pre.index("(o2-trainer-step-clear-commit-internal!")):
        raise ValueError("step successor not staged before native commit")
    step_tail = source_slice(tail, "(define (tr3-step-commit-tail!", "(define (tr3-step-commit-with-epochs-and-precommit-internal!")
    if step_tail.index("(vector-set! record 21") > step_tail.index("'commit-tail record 'step"):
        raise ValueError("step journal published before native control tail")
    restore_pre = source_slice(restore, "(define (tr3-c-joint-prepare-receiver!", "(define (tr3-c-joint-release-cell!")
    if "'stage-restore" not in restore_pre or restore_pre.index("(vector-set! parent 3 'sealed)") > restore_pre.index("'stage-restore"):
        raise ValueError("restore successor staged before authentic sealed parent")
    if restore.index("'stage-restore") > restore.index("(define (tr3-c-joint-commit-tail!"):
        raise ValueError("restore successor staged after irreversible tail")
    if core.index("'commit-tail record 'restore") > core.index("(define (tr3-lease-restore-finish-tail-internal!"):
        raise ValueError("restore journal missed genuine fixed control tail")
    route_tail = branch(route, "commit-tail")
    return_tail = branch(route, "commit-return")
    for stage in ("stage-step", "stage-restore"):
        staged = str(branch(route, stage))
        if ("['tr3-lease-nonnegative-i64?', ['vector-ref', 'row', '5']]" not in staged
                or "9223372036854775807" not in staged):
            raise ValueError("journal epoch can overflow before native commit")
    if not ("['vector-ref', 'return-cell', '0']" in str(route_tail)
            and "['tr3-lease-fail-stop', '134']" in str(route_tail)
            and "['vector-set!', ['vector-ref', 'tail', '0'], '6'" in str(route_tail)):
        raise ValueError("missing enrolled stage did not fail stop after commit")
    if "['tr3-lease-fail-stop', '134']" not in str(return_tail):
        raise ValueError("postcommit return can throw ordinary error")
    illegal = {"make-vector", "vector", "cons", "append", "raise",
               "tr3-lease-authenticate", "d2-token-dataset-cursor",
               "o2-optimizer-completed-updates-internal"}
    if any(list(descendants(route_tail, op)) or
           list(descendants(return_tail, op)) for op in illegal):
        raise ValueError("fallible or allocating operation entered fixed journal tail")
    if not all(x in wrapper for x in [
        "(g3r-x1-model-require model resolved)",
        "(g3r-d2-lineage-require dataset tokenizer)",
        "(tr3-c-snapshot-controls! record parent)",
        "(trainer-release! trainer)",
        "(tr3-controls-route 'released capsule",
        "(g3r-controls-graph-check! roots)",
        "(d2-token-dataset-cursor (vector-ref roots 5))",
        "(o2-optimizer-completed-updates-internal",
        "(tr3-lease-fail-stop 134)",
    ]):
        raise ValueError("original graph, copied controls, or failure closure changed")
    capture = source_slice(wrapper, "(define (g3r-controls-capture-release!", "(define (g3r-controls-captured-copy")
    for operation in ("step", "restore"):
        section = source_slice(wrapper,
            f"(define (g3r-controls-{operation}!",
            "(define (g3r-controls-capture-release!" if operation == "restore"
            else "(define (g3r-controls-restore!")
        if f"'commit-return capsule '{operation}" not in section:
            raise ValueError("postcommit wrapper still uses fallible leave")
        if section.rfind(f"'commit-return capsule '{operation}") < section.find("(trainer-"):
            raise ValueError("postcommit return precedes actual trainer call")
    release_stages = [capture.find(marker) for marker in (
        "(let ((trainer #f)", "(guard (caught",
        "(set! trainer (tr3-controls-route 'trainer",
        "(set! roots (tr3-controls-route 'originals")]
    if -1 in release_stages or release_stages != sorted(release_stages):
        raise ValueError("release cleanup was installed after issuer lookup")
    success = capture[capture.index("(let* ((record"):]
    if not (success.index("(tr3-c-snapshot-controls! record parent)") <
            success.index("(trainer-release! trainer)") <
            success.index("(tr3-controls-route 'released capsule") <
            success.index("(g3r-controls-graph-check! roots)") <
            success.index("(tr3-controls-route 'publish capsule")):
        raise ValueError("capture/release publication order changed")
    after_release = capture[capture.index("(tr3-controls-route 'released capsule"):]
    # The caught-failure branch may query a possibly live shell; the successful
    # postrelease path must not do so. Check the linear success subtree.
    postrelease = success[success.index("(set! released? #t)"):success.index("(tr3-controls-route 'publish capsule")]
    if "tr3-lease-record-for-shell-internal" in postrelease:
        raise ValueError("dead trainer shell consulted after release")
    if parse(aggregate) != [["load", '"g3r_g3_creation_lineage_root.esk"'],
                            ["load", '"g3r_training_controls_extension.esk"']]:
        raise ValueError("private aggregate load order changed")
    old = (ROOT / "native/g3r_g3_creation_lineage_source_closure.txt").read_text().splitlines()
    expected = ["native/g3r_training_controls_root.esk", *old,
                "native/g3r_training_controls_extension.esk"]
    if manifest.splitlines() != expected or len(expected) != len(set(expected)):
        raise ValueError("recursive private source closure changed")


def identity_probe(shell, issued):
    """Evaluate the parsed issuer lookup's exact identity predicate."""
    route = define(CORE, "tr3-controls-route")
    binding = next(x for x in route[2][2][1] if x[0] == "find-shell")
    predicate = next(x for x in descendants(binding, "eq?") if "shell" in x)
    def evaluate(node, row):
        if node == "shell":
            return shell
        if node == "rest":
            return [row]
        if node == "0":
            return 0
        if isinstance(node, list):
            op, *args = node
            if op == "car":
                return evaluate(args[0], row)[0]
            if op == "vector-ref":
                return evaluate(args[0], row)[evaluate(args[1], row)]
            if op == "eq?":
                return evaluate(args[0], row) is evaluate(args[1], row)
        raise ValueError(node)
    return any(evaluate(predicate, row) for row in issued)


def eval_route_branch(name, env, core_text=None):
    """Run the parsed guard/fixed-tail branch over owned Python cells."""
    route = (define(CORE, "tr3-controls-route") if core_text is None
             else parse(core_text)[-1])
    item = branch(route, name)
    def run(node, local):
        if isinstance(node, str):
            if node == "#t":
                return True
            if node == "#f":
                return False
            if node.isdecimal():
                return int(node)
            return local.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op == "let":
            inner = dict(local)
            for key, value in args[0]:
                inner[key] = run(value, local)
            return run_sequence(args[1:], inner)
        if op == "let*":
            inner = dict(local)
            for key, value in args[0]:
                inner[key] = run(value, inner)
            return run_sequence(args[1:], inner)
        if op == "if":
            return run(args[1], local) if run(args[0], local) else (
                run(args[2], local) if len(args) > 2 else None)
        if op == "begin":
            return run_sequence(args, local)
        if op == "or":
            return any(run(x, local) for x in args)
        if op == "and":
            return all(run(x, local) for x in args)
        if op == "=":
            return run(args[0], local) == run(args[1], local)
        if op == "not":
            return not run(args[0], local)
        if op == "eq?":
            left, right = run(args[0], local), run(args[1], local)
            return left == right if isinstance(left, str) and isinstance(right, str) else left is right
        if op == "vector-ref":
            return run(args[0], local)[run(args[1], local)]
        if op == "vector":
            return [run(x, local) for x in args]
        if op == "vector-set!":
            run(args[0], local)[run(args[1], local)] = run(args[2], local)
            return None
        if op == "find-record":
            return local.get("registered") if run(args[0], local) is local["a"] else None
        if op == "live-row":
            return local["registered_row"] if run(args[0], local) is local["a"] else None
        if op == "live-originals?":
            return local["originals_ok"]
        if op == "copy-controls":
            return local["current_controls"]
        if op == "same-controls?":
            return run(args[0], local) == run(args[1], local)
        if op == "o2-optimizer-completed-updates-internal":
            return local["o2_updates"]
        if op == "tr3-lease-record-for-shell-internal":
            return local["record"] if run(args[0], local) is local["trainer"] else None
        if op == "tr3-lease-nonnegative-i64?":
            value = run(args[0], local)
            return type(value) is int and 0 <= value <= 9223372036854775807
        if op == "clear-tail!":
            local["tail"][:] = [False, False, False, False]
            return None
        if op == "fail":
            raise ValueError("enrolled mutation denied")
        if op == "tr3-lease-fail-stop":
            raise SystemExit(134)
        raise ValueError(f"unsupported source opcode {op}")
    def run_sequence(parts, local):
        value = None
        for part in parts:
            value = run(part, local)
        return value
    return run_sequence(item[1:], env)


class CapsuleSourceTests(unittest.TestCase):
    def test_source_and_aggregate(self):
        check_source(CORE.read_text(), STEP.read_text(), TAIL.read_text(),
                     RESTORE.read_text(), WRAPPER.read_text(),
                     AGGREGATE.read_text(), MANIFEST.read_text())

    def test_issuer_exact_shell_identity(self):
        shell = ["capsule"]
        issued = [[shell]]
        self.assertTrue(identity_probe(shell, issued))
        self.assertFalse(identity_probe(["capsule"], issued))
        self.assertFalse(identity_probe(object(), issued))

    def test_guard_and_tail_transition_shapes(self):
        route = define(CORE, "tr3-controls-route")
        guard = str(branch(route, "guard"))
        tail = str(branch(route, "commit-tail"))
        for phase in ("step", "restore", "release"):
            self.assertIn("['vector-ref', 'row', '6']", guard)
            self.assertIn("['eq?', ['vector-ref', 'row', '6'], 'b']", guard)
        self.assertIn("['vector-set!', ['vector-ref', 'tail', '0'], '4', ['vector-ref', 'tail', '2']]", tail)
        self.assertIn("['vector-set!', ['vector-ref', 'tail', '0'], '5', ['vector-ref', 'tail', '3']]", tail)
        self.assertIn("['tr3-lease-fail-stop', '134']", tail)

    def test_parsed_guard_denies_direct_enrolled_step_restore_release(self):
        record, row = object(), [None] * 9
        for operation in ("step", "restore", "release"):
            row[6] = "idle"
            row[7] = False
            with self.assertRaisesRegex(ValueError, "enrolled mutation"):
                eval_route_branch("guard", {"a": record, "b": operation,
                                            "registered": row})
            row[6] = operation
            row[7] = operation
            self.assertTrue(eval_route_branch("guard", {"a": record,
                "b": operation, "registered": row}))
            self.assertFalse(row[7])
            with self.assertRaisesRegex(ValueError, "enrolled mutation"):
                eval_route_branch("guard", {"a": record, "b": operation,
                                            "registered": row})
        self.assertTrue(eval_route_branch("guard", {"a": record,
            "b": "step", "registered": None}))

    def test_parsed_enter_rejects_idle_control_drift_before_grant(self):
        shell, trainer, optimizer = object(), object(), object()
        record = [None] * 22
        record[3], record[4], record[13] = "idle", False, optimizer
        record[19:22] = [4, 3, 2]
        row = [shell, trainer, record, None, ["rng", "cursor", 4], 0,
               "idle", False, False]
        env = {"a": shell, "b": "step", "registered_row": row,
               "record": record, "trainer": trainer, "originals_ok": True,
               "current_controls": ["rng", "cursor", 4], "o2_updates": 3,
               "return-cell": [False, False, False]}
        self.assertTrue(eval_route_branch("enter", env))
        self.assertEqual((row[6], row[7]), ("step", "step"))
        # Busy-operation denial is a separate control, not evidence that a
        # changed idle journal is compared to its original baseline.
        with self.assertRaisesRegex(ValueError, "enrolled mutation"):
            eval_route_branch("enter", env)
        row[6], row[7] = "idle", False
        env["return-cell"][:] = [False, False, False]
        env["current_controls"] = ["changed", "cursor", 4]
        journal_before, epoch_before = list(row[4]), row[5]
        metrics_root = ["same staged metrics"]
        env["tr3-metrics"] = metrics_root
        with self.assertRaisesRegex(ValueError, "enrolled mutation"):
            eval_route_branch("enter", env)
        self.assertEqual((row[6], row[7]), ("idle", False))
        self.assertEqual(row[4], journal_before)
        self.assertEqual(row[5], epoch_before)
        self.assertEqual(metrics_root, ["same staged metrics"])
        self.assertEqual(env["return-cell"], [False, False, False])
        mutated = CORE.read_text().replace(
            "(not (same-controls? (vector-ref row 4) current))", "#f", 1)
        self.assertTrue(eval_route_branch("enter", env, mutated))
        self.assertEqual((row[6], row[7]), ("step", "step"))

    def test_parsed_step_and_restore_tail_publish_exact_staged_epoch(self):
        record = object()
        for operation in ("step", "restore"):
            shell = object()
            row = [shell, None, record, None, "old", 4, operation, False, False]
            successor = [operation, 5]
            tail = [row, record, successor, 5]
            returning = [row, shell, operation]
            env = {"a": record, "b": operation, "tail": tail,
                   "return-cell": returning}
            self.assertTrue(eval_route_branch("commit-tail", env))
            self.assertIs(row[4], successor)
            self.assertEqual(row[5], 5)
            self.assertEqual(row[6], "committed-" + operation)
            self.assertEqual(tail, [False] * 4)
            record = [None] * 5
            record[3], record[4] = "idle", False
            row[2] = record
            self.assertTrue(eval_route_branch("commit-return", {
                "a": shell, "b": operation, "tail": tail,
                "return-cell": returning}))
            self.assertEqual(row[6], "idle")
            self.assertEqual(returning, [False] * 3)
            row[2] = record = object()
            row[6] = operation
            returning[:] = [row, shell, operation]
            with self.assertRaises(SystemExit) as missing:
                eval_route_branch("commit-tail", dict(env, a=record,
                    tail=[False] * 4))
            self.assertEqual(missing.exception.code, 134)
            wrong = [row, object(), successor, 6]
            with self.assertRaises(SystemExit) as failure:
                eval_route_branch("commit-tail", dict(env, a=record, tail=wrong))
            self.assertEqual(failure.exception.code, 134)
            self.assertTrue(eval_route_branch("commit-tail", {
                "a": object(), "b": operation, "tail": [False] * 4,
                "return-cell": [False] * 3}))

    def test_parsed_postcommit_divergence_fails_stop(self):
        shell = object()
        record = [None] * 5
        record[3], record[4] = "committing", False
        row = [shell, None, record, None, "staged", 1,
               "committed-step", False, False]
        with self.assertRaises(SystemExit) as failure:
            eval_route_branch("commit-return", {"a": shell, "b": "step",
                "tail": [False] * 4, "return-cell": [row, shell, "step"]})
        self.assertEqual(failure.exception.code, 134)
        self.assertEqual(row[6], "committed-step")

    def test_mutant_missing_step_gate(self):
        changed = STEP.read_text().replace("(tr3-controls-route 'guard record 'step #f #f #f)", "#t", 1)
        with self.assertRaisesRegex(ValueError, "guard"):
            check_source(CORE.read_text(), changed, TAIL.read_text(), RESTORE.read_text(),
                         WRAPPER.read_text(), AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_train_gate_after_metrics(self):
        train = TRAIN.read_text()
        changed = train.replace("(tr3-controls-route 'guard record 'train #f #f #f)", "#t", 1)
        with self.assertRaisesRegex(ValueError, "metrics mutate"):
            check_train_source(changed)

    def test_mutant_enter_without_prior_journal_check(self):
        changed = CORE.read_text().replace(
            "(not (same-controls? (vector-ref row 4) current))", "#f", 1)
        with self.assertRaisesRegex(ValueError, "prior control journal"):
            check_source(changed, STEP.read_text(), TAIL.read_text(),
                         RESTORE.read_text(), WRAPPER.read_text(),
                         AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_reusable_same_phase_grant(self):
        changed = CORE.read_text().replace(
            "(vector-set! row 7 #f)))\n             #t))", "#t))\n             #t))", 1)
        with self.assertRaisesRegex(ValueError, "same-phase"):
            check_source(changed, STEP.read_text(), TAIL.read_text(),
                         RESTORE.read_text(), WRAPPER.read_text(),
                         AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_tail_allocation(self):
        changed = CORE.read_text().replace("(clear-tail!)))", "(cons a b) (clear-tail!)))", 1)
        with self.assertRaisesRegex(ValueError, "fixed journal tail"):
            check_source(changed, STEP.read_text(), TAIL.read_text(), RESTORE.read_text(),
                         WRAPPER.read_text(), AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_trainer_lookup_gap(self):
        changed = CORE.read_text().replace("(vector-set! row 1 trainer)\n                 (let*", "(let*", 1)
        with self.assertRaisesRegex(ValueError, "cleanup root"):
            check_source(changed, STEP.read_text(), TAIL.read_text(), RESTORE.read_text(),
                         WRAPPER.read_text(), AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_missing_original_handle_copy(self):
        changed = CORE.read_text().replace("(vector-copy (vector-ref record 11))", "(vector-ref record 11)", 1)
        with self.assertRaisesRegex(ValueError, "14-handle"):
            check_source(changed, STEP.read_text(), TAIL.read_text(), RESTORE.read_text(),
                         WRAPPER.read_text(), AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_restore_staged_after_tail(self):
        changed = RESTORE.read_text().replace("(tr3-controls-route 'stage-restore record parent #f #f #f)", "#t", 1)
        with self.assertRaisesRegex(ValueError, "restore successor"):
            check_source(CORE.read_text(), STEP.read_text(), TAIL.read_text(), changed,
                         WRAPPER.read_text(), AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_release_before_control_copy(self):
        changed = WRAPPER.read_text().replace(
            "(controls (tr3-c-snapshot-controls! record parent))",
            "(controls (begin (trainer-release! trainer)\n"
            "                    (tr3-c-snapshot-controls! record parent)))", 1)
        with self.assertRaisesRegex(ValueError, "publication order"):
            check_source(CORE.read_text(), STEP.read_text(), TAIL.read_text(),
                         RESTORE.read_text(), changed, AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_lookup_before_release_cleanup(self):
        changed = WRAPPER.read_text().replace(
            "(let ((trainer #f)\n        (roots #f)",
            "(let ((trainer (tr3-controls-route 'trainer capsule #f #f #f #f))\n"
            "        (roots #f)", 1)
        with self.assertRaisesRegex(ValueError, "release cleanup"):
            check_source(CORE.read_text(), STEP.read_text(), TAIL.read_text(),
                         RESTORE.read_text(), changed, AGGREGATE.read_text(), MANIFEST.read_text())

    def test_mutant_postrelease_dead_shell_query(self):
        changed = WRAPPER.read_text().replace("(g3r-controls-graph-check! roots)\n        (if (or", "(tr3-lease-record-for-shell-internal trainer 'bad)\n        (g3r-controls-graph-check! roots)\n        (if (or", 1)
        with self.assertRaisesRegex(ValueError, "dead trainer"):
            check_source(CORE.read_text(), STEP.read_text(), TAIL.read_text(),
                         RESTORE.read_text(), changed, AGGREGATE.read_text(), MANIFEST.read_text())


if __name__ == "__main__":
    unittest.main()
