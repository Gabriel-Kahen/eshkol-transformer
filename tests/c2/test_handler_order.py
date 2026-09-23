"""Source-order regression checks; these do not prove runtime allocation recovery."""
from copy import deepcopy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
P1 = "templates/p1/module_roots.esk.tmpl"
C1 = "internal/c1/lib/transformer/checkpoint_internal.esk"
E1 = "lib/transformer/error_core.esk"
FILES = (P1, "internal/p1/lib/transformer/module.esk", *(f"native/{name}.esk" for name in (
    "c2_training_state_extension", "c2_checkpoint_save_extension",
    "c2_o2_reconstruct_extension", "c2_checkpoint_load_extension", "c2_public_extension")),
    C1, E1)


def parse(source):
    """Small datum reader: strings/escapes, line/block/datum comments and quote."""
    tokens, i = [], 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch == ";":
            end = source.find("\n", i)
            i = len(source) if end < 0 else end + 1
        elif source.startswith("#|", i):
            depth, i = 1, i + 2
            while depth and i < len(source):
                if source.startswith("#|", i):
                    depth, i = depth + 1, i + 2
                elif source.startswith("|#", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                raise ValueError("unterminated block comment")
        elif source.startswith("#;", i):
            tokens.append("#;")
            i += 2
        elif ch == '"':
            start, i = i, i + 1
            while i < len(source) and source[i] != '"':
                i += 2 if source[i] == "\\" else 1
            if i >= len(source):
                raise ValueError("unterminated string")
            i += 1
            tokens.append(source[start:i])
        elif source.startswith("#\\", i):
            start, i = i, i + 2
            if i >= len(source):
                raise ValueError("missing character")
            i += 1
            while i < len(source) and not source[i].isspace() and source[i] not in "()":
                i += 1
            tokens.append(source[start:i])
        elif ch in "()'`,":
            token, i = ch, i + 1
            if token == "," and i < len(source) and source[i] == "@":
                token, i = ",@", i + 1
            tokens.append(token)
        else:
            start = i
            while i < len(source) and not source[i].isspace() and source[i] not in "();'`\",":
                i += 1
            tokens.append(source[start:i])
    position = 0

    def datum():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("missing datum")
        token, position = tokens[position], position + 1
        if token == "(":
            result = []
            while position < len(tokens) and tokens[position] != ")":
                if tokens[position] == "#;":
                    position += 1
                    datum()
                else:
                    result.append(datum())
            if position >= len(tokens):
                raise ValueError("unclosed list")
            position += 1
            return result
        if token == "#;":
            datum()
            return datum()
        if token == ")":
            raise ValueError("unexpected close")
        if token in ("'", "`", ",", ",@"):
            return [{"'": "quote", "`": "quasiquote", ",": "unquote", ",@": "unquote-splicing"}[token], datum()]
        return token

    result = []
    while position < len(tokens):
        if tokens[position] == "#;":
            position += 1
            datum()
        else:
            result.append(datum())
    return result


def form(text):
    return parse(text)[0]


def render(node):
    return "(" + " ".join(map(render, node)) + ")" if isinstance(node, list) else node


def need(condition, message):
    if not condition:
        raise ValueError(message)


def lists(node):
    if isinstance(node, list):
        yield node
        if node[:1] not in (["quote"], ["quasiquote"]):
            for child in node:
                yield from lists(child)


def definition(sources, name, path=None):
    found = [node for key, roots in sources.items() if path is None or key == path
             for node in lists(roots) if node[:1] == ["define"]
             and len(node) > 2 and isinstance(node[1], list) and node[1][:1] == [name]]
    need(len(found) == 1, f"expected one definition: {name}")
    return found[0]


def implementation_definition(sources, name, path, effect):
    found = [node for node in lists(sources[path]) if node[:1] == ["define"]
             and len(node) > 2 and isinstance(node[1], list)
             and node[1][:1] == [name] and matches(node, effect)]
    need(len(found) == 1, f"expected one implementation definition: {name}")
    return found[0]


def events(node, guards=()):
    """Executable AST nodes with ACTIVE guard bodies; no quoted/lambda decoys.

    Initializers execute outside their enclosing let body but under its active
    guards. A guard's own handler is deliberately not protected by that guard.
    """
    if not isinstance(node, list) or not node:
        return
    head = node[0]
    if head in ("quote", "quasiquote", "lambda", "define-syntax"):
        return
    yield node, guards
    if head == "guard":
        for child in node[2:]:
            yield from events(child, (*guards, node))
    elif head == "define":
        for child in node[2:]:
            yield from events(child, guards)
    elif head in ("let", "let*", "letrec"):
        index = 2 if isinstance(node[1], str) else 1
        for binding in node[index]:
            yield from events(binding[1], guards)
        for child in node[index + 1:]:
            yield from events(child, guards)
    elif head == "with-region":
        for child in node[2:]:
            yield from events(child, guards)
    else:
        for child in node[1:]:
            yield from events(child, guards)


def matches(node, text):
    wanted = form(text)
    return [(item, guards) for item, guards in events(node) if item == wanted]


def unique(node, text):
    found = matches(node, text)
    need(len(found) == 1, f"expected one executable {text}, found {len(found)}")
    return found[0]


def contains(node, text):
    need(matches(node, text), f"missing executable {text}")


def handler(guard):
    return ["begin", *(expression for clause in guard[1][1:] for expression in clause[1:])]


def protected(node, text, depth, cleanup=None):
    item, guards = unique(node, text)
    need(len(guards) == depth, f"wrong active guard depth for {text}")
    if cleanup:
        contains(handler(guards[-1]), cleanup)
    return guards


def consecutive(node, *texts):
    wanted = [form(text) for text in texts]
    need(any(any(parent[i:i + len(wanted)] == wanted
                 for i in range(len(parent))) for parent, _ in events(node)),
         f"missing consecutive executable prefix: {texts}")


def preallocated(node, binding_text, guard):
    """Binding must be available before setup, including let* initializer order."""
    binding = form(binding_text)
    for parent, _ in events(node):
        if parent[0] not in ("let", "let*", "letrec"):
            continue
        index = 2 if isinstance(parent[1], str) else 1
        if binding not in parent[index]:
            continue
        for sibling_index, sibling in enumerate(parent[index]):
            if any(item is guard for item, _ in events(sibling[1])):
                need(parent[0] == "let*" and parent[index].index(binding) < sibling_index,
                     f"bookkeeping initializer is too late: {binding_text}")
                return
        if any(item is guard for child in parent[index + 1:] for item, _ in events(child)):
            return
    raise ValueError(f"bookkeeping must precede handler setup: {binding_text}")


def p1_release_body(sources, path, claimed=False):
    parameters = ["state", "owner-token"] if claimed else ["state"]
    found = [["begin", *node[2:]] for node in lists(sources[path])
             if node[:2] == ["lambda", parameters]
             and matches(["begin", *node[2:]],
                         "(p1-native-state-release-begin native-context state)")]
    need(len(found) == 1, "expected one P1 release boundary")
    return found[0]


def reserve_before(node, name, count, targets):
    """Require a checked reserve in an earlier statement, not a branch decoy."""
    calls = [item for item, _ in events(node) if item[:1] == [name]]
    need(len(calls) == 1 and calls[0] == [name, str(count)],
         f"expected one literal {name}/{count} reserve")
    checked = [item for item, _ in events(node)
               if item[:2] == ["if", ["not", ["=", calls[0], "0"]]]]
    need(len(checked) == 1, f"reserve status must be checked: {name}")
    check = checked[0]
    for target in targets:
        ordered = False
        for parent, _ in events(node):
            if parent[0] in ("begin", "define", "guard"):
                body = parent[1:] if parent[0] == "begin" else parent[2:]
            elif parent[0] in ("let", "let*"):
                body = parent[3:] if isinstance(parent[1], str) else parent[2:]
            else:
                continue
            for index, statement in enumerate(body):
                if statement is check and any(
                    item is target for later in body[index + 1:]
                    for item, _ in events(later)
                ):
                    ordered = True
        need(ordered, f"{name} must precede guard/effect: {render(target)}")
    return check


RESERVES = (
    ("p1-public", "p1-runtime-reserve-exception-handlers", 5,
     "(p1-native-state-release-begin native-context state)"),
    ("c2-training-state-release-internal!", "c2-runtime-reserve-exception-handlers", 6,
     "(c2-training-state-release-take-authority! record value #f operation)"),
    ("c2-checkpoint-load-reconstruct-internal", "c2-runtime-reserve-exception-handlers", 11,
     "(vector-set! entry 2 'consuming)"),
)
RESERVE_DECLARATIONS = (
    (P1, "p1-runtime-reserve-exception-handlers"),
    (FILES[1], "p1-runtime-reserve-exception-handlers"),
    ("native/c2_training_state_extension.esk", "c2-runtime-reserve-exception-handlers"),
)


def reserve_site(sources, site, p1_path=P1):
    return (p1_release_body(sources, p1_path) if site == "p1-public"
            else definition(sources, site))


def reserve_targets(node, site, effect):
    mutation, guards = unique(node, effect)
    if site == "p1-public":
        guards = unique(node, "(state-dict-release-owned! raw release-outcome)")[1]
    targets = [mutation, *guards]
    if site == "c2-checkpoint-load-reconstruct-internal":
        targets.append(unique(node, "(c2-load-decode-model-internal model subpolicy)")[0])
    return targets


def lifecycle_constructors(sources, path):
    return [node for node in lists(sources[path])
            if node[:2] == ["vector", "state-lifecycle-tag"]]


def validate_p1_scratch(sources):
    constructors = lifecycle_constructors(sources, P1)
    need(len(constructors) == 3, "expected all three P1 lifecycle constructors")
    need(constructors == lifecycle_constructors(sources, FILES[1]),
         "generated lifecycle constructors differ")
    for constructor in constructors:
        need(len(constructor) == 11
             and constructor[9:] == [form("(vector 'empty #f #f #f)"),
                                     form("(vector #f #f)")],
             "lifecycle requires exact length10 and scratch slots8/9")
    contains(definition(sources, "require-live-state", P1),
             "(= (vector-length lifecycle) 10)")
    for path in (P1, FILES[1]):
        callee = definition(sources, "state-dict-release-owned!", path)
        need(callee[1] == ["state-dict-release-owned!", "state", "release-outcome"],
             "P1 release must receive preallocated scratch")
        for owner in (callee, p1_release_body(sources, path),
                      p1_release_body(sources, path, claimed=True)):
            need(not any(item[:1] == ["vector"] for item, _ in events(owner)),
                 "P1 release must not allocate scratch vectors")
        for claimed in (False, True):
            owner = p1_release_body(sources, path, claimed=claimed)
            lifecycle = "lifecycle" if claimed else "release-lifecycle"
            bindings = [form(f"({lifecycle} (state-lifecycle raw))"),
                        form(f"(release-outcome (vector-ref {lifecycle} 8))")]
            if not claimed:
                bindings.append(form("(outcome (vector-ref release-lifecycle 9))"))
            need(any(item[:1] == ["let*"] and item[1][:len(bindings)] == bindings
                     for item, _ in events(owner)),
                 "release scratch must come from canonical lifecycle slots")
            targets = [unique(owner, "(p1-native-state-release-begin native-context state)")[0],
                       unique(owner, "(compact-released-state-shells! raw)")[0]]
            call, guards = unique(owner, "(state-dict-release-owned! raw release-outcome)")
            targets.extend([call, *guards])
            if claimed:
                targets.append(unique(owner, "(vector-set! lifecycle 6 #f)")[0])
            else:
                targets.append(unique(owner, "(p1-runtime-reserve-exception-handlers 5)")[0])
            for binding in bindings:
                for target in targets:
                    preallocated(owner, render(binding), target)


# These effects are also independently moved outside protection by mutation tests.
TARGETS = (
    ("state-dict-release-owned!", P1, "(vector-set! lifecycle 1 'released)", 1),
    ("c2-training-state-compose-internal", None, "(c2-training-state-claim-p1! p1 owner-token)", 1),
    ("c2-training-state-borrow-begin-internal", None, "(vector-set! state 7 borrow)", 1),
    ("c2-training-state-borrow-begin-internal", None,
     "(c2-training-state-activate-p1-borrow! (vector-ref state 1) active-owner-token active-access-token active-borrow operation)", 2),
    ("c2-save-with-owner-borrow", None, "(c2-save-borrow-begin-internal state 'checkpoint-save!)", 1),
    ("o2-state-reconstruct-internal", None,
     "(o2-c2-native-reconstruct-create-internal (vector-length entries) (vector-ref clip 0) (vector-ref clip 1) (vector-ref schedule 0) (vector-ref schedule 1) (vector-ref schedule 2) (vector-ref schedule 3) completed)", 1),
    ("c2-training-state-release-internal!", None,
     "(c2-training-state-release-take-authority! record value #f operation)", 2),
    ("c2-public-checkpoint-load", None, "(c2-checkpoint-load-stage-internal path policy)", 1),
    ("c2-checkpoint-load-reconstruct-internal", None, "(vector-set! entry 2 'consuming)", 6),
    ("c2-checkpoint-load-reconstruct-internal", None, "(c2-load-decode-model-internal model subpolicy)", 6),
    ("c2-checkpoint-load-reconstruct-internal", None, "(c2-load-release-p1-internal p1)", 5),
    ("c2-checkpoint-load-reconstruct-internal", None, "(c2-load-release-o2-internal o2)", 4),
    ("c2-checkpoint-load-reconstruct-internal", None,
     "(c2-load-release-c2-take-internal! c2-release-record c2 c2-cell 'trainer-state-release!)", 3),
)


def validate(sources):
    validate_p1_scratch(sources)
    for path, name in RESERVE_DECLARATIONS:
        declarations = [item for item in sources[path]
                        if item[:1] == ["extern"] and len(item) > 2 and item[2] == name]
        need(declarations == [["extern", "i64", name, "i64", ":real",
                               "eshkol_runtime_reserve_exception_handlers_v1"]],
             f"wrong reserve ABI: {path}/{name}")
    for site, name, count, effect in RESERVES:
        node = reserve_site(sources, site)
        reserve_before(node, name, count, reserve_targets(node, site, effect))
    need(p1_release_body(sources, P1) == p1_release_body(sources, FILES[1]),
         "generated public P1 release differs")
    claimed = p1_release_body(sources, P1, claimed=True)
    need(claimed == p1_release_body(sources, FILES[1], claimed=True),
         "generated claimed P1 release differs")
    need(not any(item[:1] == ["p1-runtime-reserve-exception-handlers"]
                 for item, _ in events(claimed)),
         "claimed P1 release relies on enclosing C2/LOAD reserve")
    for name, path, effect, depth in TARGETS:
        protected(definition(sources, name, path), effect, depth)
    for name, path, binding, effect in (
        ("c2-training-state-compose-internal", None, "(claim-progress (vector #f))", TARGETS[1][2]),
        ("c2-training-state-borrow-begin-internal", None, "(p1-active (vector #f))", TARGETS[2][2]),
        ("c2-save-with-owner-borrow", None, "(transaction (vector #f #f #f #f))", TARGETS[4][2]),
        ("o2-state-reconstruct-internal", None, "(builder-cell (vector #f))", TARGETS[5][2]),
        ("c2-training-state-release-internal!", None, "(record (c2-training-state-release-record value))", TARGETS[6][2]),
        ("c2-public-checkpoint-load", None, "(stage-cell (vector #f))", TARGETS[7][2]),
        ("c2-checkpoint-load-reconstruct-internal", None, "(c2-release-record (c2-training-state-release-record #f))", TARGETS[8][2]),
    ):
        node = definition(sources, name, path)
        preallocated(node, binding, unique(node, effect)[1][0])
    p1 = definition(sources, "state-dict-release-owned!", P1)
    need(p1 == definition(sources, "state-dict-release-owned!", FILES[1]), "generated P1 differs")
    for effect in ("(vector-set! lifecycle 4 '())", "(vector-set! ownership 1 '())",
                   "(vector-set! (car entries) 7 #f)",
                   "(release-owned-list! 'state-dict-release! provider ownership)"):
        protected(p1, effect, 1)
    consecutive(p1, "(vector-set! release-outcome 3 lifecycle)",
                "(vector-set! release-outcome 0 'started)", "(vector-set! lifecycle 1 'released)")
    contains(handler(unique(p1, "(vector-set! lifecycle 1 'released)")[1][-1]),
             "(if (not (eq? (vector-ref release-outcome 0) 'started)) (raise caught))")

    compose = definition(sources, "c2-training-state-compose-internal")
    guards = protected(compose, "(c2-training-state-claim-p1! p1 owner-token)", 1,
                       "(c2-training-state-rollback-p1! p1 owner-token)")
    consecutive(guards[-1], "(c2-training-state-claim-p1! p1 owner-token)",
                "(vector-set! claim-progress 0 #t)", "(c2-training-state-claim-o2! o2 owner-token)")
    contains(handler(guards[-1]), "(if (vector-ref claim-progress 0) (begin (c2-training-state-rollback-p1! p1 owner-token) (vector-set! claim-progress 0 #f)))")
    borrow = definition(sources, "c2-training-state-borrow-begin-internal")
    call = "(c2-training-state-activate-p1-borrow! (vector-ref state 1) active-owner-token active-access-token active-borrow operation)"
    guards = unique(borrow, call)[1]
    for guard in guards:
        contains(handler(guard), "(c2-training-state-borrow-activation-rollback! state p1-active operation)")
        contains(handler(guard), "(raise caught)")
    consecutive(guards[-1], call, "(vector-set! p1-active 0 #t)",
                "(c2-training-state-activate-o2-borrow! (vector-ref state 2) active-owner-token active-access-token active-borrow operation)")
    need(any(node[:2] == ["let*", form("((active-borrow (vector-ref state 7)) (active-owner-token (vector-ref active-borrow 3)) (active-access-token (vector-ref active-borrow 4)))")]
             for node, _ in events(borrow)), "missing canonical borrow readback")
    rollback = definition(sources, "c2-training-state-borrow-activation-rollback!")
    for text in ("(vector-ref state 6)", "(vector-ref state 7)",
                 "(state-dict-c2-owned-borrow-deactivate-internal! (vector-ref state 1) owner-token (vector-ref active-borrow 4) active-borrow operation)",
                 "(vector-set! p1-active 0 #f)", "(vector-set! owner-token 2 #f)",
                 "(vector-set! state 7 'live)"):
        contains(rollback, text)

    save = definition(sources, "c2-save-with-owner-borrow")
    need(save[2][0] == "let*" and save[2][1][0] == form("(transaction (vector #f #f #f #f))"),
         "SAVE transaction must precede guarded borrow initializer")
    protected(save, "(vector-set! transaction 0 borrow)", 1)
    contains(save, "(if (and (vector-ref transaction 0) (not (vector-ref transaction 1))) (begin (c2-training-state-borrow-end-internal (vector-ref transaction 0)) (vector-set! transaction 1 #t)))")
    builder = definition(sources, "o2-state-reconstruct-internal")
    guard = unique(builder, TARGETS[5][2])[1][-1]
    consecutive(builder, "(vector-set! builder-cell 0 builder)", "(o2-c2-native-fill! operation builder entries)")
    contains(handler(guard), "(o2-c2-native-reconstruct-abort builder)")
    contains(handler(guard), "(vector-set! builder-cell 0 #f)")
    consecutive(builder, "(vector-set! builder-cell 0 #f)", "(vector-set! state 1 native)")

    release = definition(sources, "c2-training-state-release-internal!")
    for guard in unique(release, TARGETS[6][2])[1]:
        contains(handler(guard), "(c2-training-state-release-record-note-defect! record defect caught)")
    protected(release, "(c2-training-state-release-record-p1! record operation)", 2)
    protected(release, "(c2-training-state-release-record-o2! record operation)", 1)
    protected(release, "(c2-training-state-release-record-finalize! record operation)", 0)
    contains(definition(sources, "c2-training-state-release-take-authority!"),
             "(vector-set! record 2 'started)")
    contains(definition(sources, "c2-training-state-release-record-note-defect!"),
             "(if (not (eq? (vector-ref record 2) 'started)) (raise caught))")
    public = definition(sources, "c2-public-checkpoint-load")
    protected(public, "(c2-checkpoint-load-stage-internal path policy)", 1,
              "(c2-checkpoint-load-staging-release-internal! stage)")
    protected(public, "(vector-set! stage-cell 0 stage)", 1)
    contains(public, "(c2-checkpoint-load-staging-dead-internal? (vector-ref stage-cell 0))")
    load = definition(sources, "c2-checkpoint-load-reconstruct-internal")
    guards = unique(load, "(vector-set! entry 2 'consuming)")[1]
    for guard in guards[:-1]:
        contains(handler(guard), "(c2-load-note-cleanup-defect-or-rethrow! entry cleanup-outcome caught)")
    contains(definition(sources, "c2-load-note-cleanup-defect-or-rethrow!"),
             "(if (eq? (vector-ref entry 2) 'live) (raise caught))")
    for text, depth in (("(vector-set! p1-cell 0 #f)", 5), ("(vector-set! o2-cell 0 #f)", 4),
                        ("(c2-load-release-c2-p1-internal! c2-release-record 'trainer-state-release!)", 3),
                        ("(c2-load-release-c2-o2-internal! c2-release-record 'trainer-state-release!)", 2),
                        ("(c2-load-release-c2-finalize-internal! c2-release-record 'trainer-state-release!)", 1)):
        protected(load, text, depth)
    # LOAD's eleven-frame maximum is a reachable dual-fault path: its six
    # guards enclose C1 ownership, then P1 adoption cleanup encloses item,
    # provider callback, and E1 shell-construction guards. These source seams
    # are fixed independently so a sequential guard cannot be miscounted.
    c1 = definition(sources, "c1-checkpoint-decode-state-internal", C1)
    protected(c1,
              "(state-dict-adopt-owned-internal! ordered aliases provider-name 2 0 ownership)",
              1)
    adopt = implementation_definition(
        sources, "state-dict-adopt-owned-internal!", P1,
        "(release-owned-list! operation provider adopted-ownership)")
    cleanup = protected(adopt,
                        "(release-owned-list! operation provider adopted-ownership)", 1)
    consecutive(adopt, "(vector-set! ownership 0 '())",
                render(unique(adopt,
                              "(validate-state-with-provider operation state provider)")[1][-1]))
    need(cleanup[-1] is not unique(adopt,
                                   "(native-check operation (p1-native-state-revoke native-context state-shell))")[1][-1],
         "P1 adoption release and revoke guards must remain sequential")
    protected(definition(sources, "make-shell", E1),
              '(error "transformer.error_internal:v1")', 1)


class HandlerOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {path: parse((ROOT / path).read_text()) for path in FILES}

    def test_source_union(self):
        validate(self.sources)

    def test_reader_and_formatting(self):
        source = '''; (fake)\n(define (sample) "escaped \\\" ; (guard)" '(guard ignored)
                    #| outer #| inner |# comment |# #;(guard ignored) #\\()'''
        self.assertEqual(parse(source), parse(render(parse(source)[0])))
        self.assertEqual(parse("(a #; #; b c d) '#;x y"),
                         [["a", "d"], ["quote", "y"]])
        self.assertEqual(len(list(events(parse(source)[0]))), 1)
        formatted = {key: parse("\n; harmless (guard fake)\n".join(map(render, roots)))
                     for key, roots in self.sources.items()}
        validate(formatted)

    def test_each_effect_requires_active_guard_not_sibling(self):
        for name, path, effect, _ in TARGETS:
            for mutation in ("outside", "sibling", "quoted-decoy", "lambda-decoy"):
                with self.subTest(site=name, effect=effect, mutation=mutation):
                    changed = deepcopy(self.sources)
                    node, guards = unique(definition(changed, name, path), effect)
                    guard = guards[-1]
                    if mutation == "outside":
                        moved = deepcopy(node)
                        node[:] = ["begin", "#f"]
                        guard[:] = ["begin", moved, deepcopy(guard)]
                    elif mutation == "sibling":
                        guard[:] = ["begin", ["guard", deepcopy(guard[1]), "#t"], *guard[2:]]
                    else:
                        node[:] = (["quote", deepcopy(node)] if mutation == "quoted-decoy"
                                   else ["lambda", [], deepcopy(node)])
                    with self.assertRaises(ValueError):
                        validate(changed)

    def test_bookkeeping_cannot_follow_protected_acquisition(self):
        changed = deepcopy(self.sources)
        save = definition(changed, "c2-save-with-owner-borrow")
        save[2][1].reverse()
        with self.assertRaises(ValueError):
            validate(changed)

    def test_reserve_removal_count_and_late_placement(self):
        for site, name, count, effect in RESERVES:
            for mutation in ("removed", "count", "after-guard", "after-mutation"):
                with self.subTest(site=site, mutation=mutation):
                    changed = deepcopy(self.sources)
                    # Mutate both P1 copies so parity cannot mask an order bug.
                    paths = (P1, FILES[1]) if site == "p1-public" else (P1,)
                    for path in paths:
                        owner = reserve_site(changed, site, path)
                        targets = reserve_targets(owner, site, effect)
                        check = reserve_before(owner, name, count, targets)
                        if mutation == "count":
                            unique(owner, f"({name} {count})")[0][1] = str(count + 1)
                            continue
                        moved = deepcopy(check)
                        check[:] = ["begin", "#t"]
                        if mutation == "after-guard":
                            targets[1].insert(2, moved)
                        elif mutation == "after-mutation":
                            target = targets[0]
                            target[:] = ["begin", deepcopy(target), moved]
                    with self.assertRaisesRegex(ValueError, name):
                        validate(changed)

    def test_reserve_abi_and_claimed_dependency(self):
        for path, name in RESERVE_DECLARATIONS:
            with self.subTest(path=path):
                changed = deepcopy(self.sources)
                declaration = next(item for item in changed[path]
                                   if item[:3] == ["extern", "i64", name])
                declaration[-1] = "wrong_runtime_symbol"
                with self.assertRaisesRegex(ValueError, "reserve ABI"):
                    validate(changed)
        changed = deepcopy(self.sources)
        for path in (P1, FILES[1]):
            owner = p1_release_body(changed, path, claimed=True)
            call, _ = unique(owner, "(p1-native-state-release-begin native-context state)")
            call[:] = ["begin", ["p1-runtime-reserve-exception-handlers", "5"],
                       deepcopy(call)]
        with self.assertRaisesRegex(ValueError, "enclosing C2/LOAD reserve"):
            validate(changed)

    def test_load_dual_fault_high_water_seams(self):
        mutations = (
            (C1, "c1-checkpoint-decode-state-internal",
             "(state-dict-adopt-owned-internal! ordered aliases provider-name 2 0 ownership)"),
            (P1, "state-dict-adopt-owned-internal!",
             "(release-owned-list! operation provider adopted-ownership)"),
            (E1, "make-shell", '(error "transformer.error_internal:v1")'),
        )
        for path, name, effect in mutations:
            with self.subTest(path=path, name=name):
                changed = deepcopy(self.sources)
                node = (implementation_definition(changed, name, path, effect)
                        if name == "state-dict-adopt-owned-internal!"
                        else definition(changed, name, path))
                target, guards = unique(node, effect)
                guard = guards[-1]
                guard[:] = ["begin", *guard[2:]]
                with self.assertRaises(ValueError):
                    validate(changed)

    def test_each_lifecycle_constructor_requires_both_scratch_slots(self):
        for index in range(3):
            for mutation in ("slot8", "slot9", "length"):
                with self.subTest(constructor=index, mutation=mutation):
                    changed = deepcopy(self.sources)
                    for path in (P1, FILES[1]):
                        constructor = lifecycle_constructors(changed, path)[index]
                        if mutation == "length":
                            constructor.append("#f")
                        else:
                            constructor[9 if mutation == "slot8" else 10] = "#f"
                    with self.assertRaisesRegex(ValueError, "scratch slots8/9"):
                        validate(changed)

    def test_release_cannot_reallocate_lifecycle_scratch(self):
        for site in ("callee", "public", "claimed"):
            with self.subTest(site=site):
                changed = deepcopy(self.sources)
                for path in (P1, FILES[1]):
                    owner = (definition(changed, "state-dict-release-owned!", path)
                             if site == "callee" else
                             p1_release_body(changed, path, claimed=site == "claimed"))
                    first = owner[2 if site == "callee" else 1]
                    first[:] = ["begin", form("(vector 'empty #f #f #f)"),
                                deepcopy(first)]
                with self.assertRaisesRegex(ValueError, "must not allocate scratch"):
                    validate(changed)

    def test_canonical_scratch_slots_and_pre_native_order(self):
        for claimed, binding in ((False, "release-outcome"), (False, "outcome"),
                                 (True, "release-outcome")):
            for mutation in ("wrong-slot", "after-native"):
                with self.subTest(claimed=claimed, binding=binding, mutation=mutation):
                    changed = deepcopy(self.sources)
                    for path in (P1, FILES[1]):
                        owner = p1_release_body(changed, path, claimed=claimed)
                        holder = next(item for item, _ in events(owner)
                                      if item[:1] == ["let*"]
                                      and any(value[0] == binding for value in item[1]))
                        scratch = next(value for value in holder[1] if value[0] == binding)
                        if mutation == "wrong-slot":
                            scratch[1][-1] = "7"
                        else:
                            holder[1].remove(scratch)
                            native, _ = unique(owner,
                                "(p1-native-state-release-begin native-context state)")
                            native[:] = ["begin", deepcopy(native),
                                         ["let*", [scratch], "#t"]]
                    with self.assertRaisesRegex(ValueError, "canonical lifecycle slots"):
                        validate(changed)

    def test_phase_and_canonical_cleanup_mutations(self):
        cases = (
            ("c2-training-state-compose-internal", "(vector-set! claim-progress 0 #t)"),
            ("c2-training-state-borrow-begin-internal", "(vector-set! p1-active 0 #t)"),
            ("c2-training-state-borrow-begin-internal", "(vector-ref state 7)"),
            ("c2-training-state-borrow-activation-rollback!", "(vector-set! owner-token 2 #f)"),
            ("c2-training-state-borrow-activation-rollback!", "(vector-set! state 7 'live)"),
            ("c2-save-with-owner-borrow", "(vector-set! transaction 0 borrow)"),
            ("c2-public-checkpoint-load", "(vector-set! stage-cell 0 stage)"),
            ("c2-training-state-release-take-authority!", "(vector-set! record 2 'started)"),
            ("c2-load-note-cleanup-defect-or-rethrow!", "(if (eq? (vector-ref entry 2) 'live) (raise caught))"),
        )
        for name, effect in cases:
            with self.subTest(site=name, missing=effect):
                changed = deepcopy(self.sources)
                owner = definition(changed, name)
                if effect == "(vector-ref state 7)":
                    binding = next(node for node, _ in events(owner)
                                   if node[:1] == ["let*"] and node[1][0][0] == "active-borrow")
                    node = binding[1][0][1]
                else:
                    node, _ = unique(owner, effect)
                node[:] = ["begin", "#f"]
                with self.assertRaises(ValueError):
                    validate(changed)


if __name__ == "__main__":
    unittest.main()
