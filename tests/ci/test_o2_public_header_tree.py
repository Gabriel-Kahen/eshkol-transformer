"""Run O2's existing public-surface pin during early CI discovery."""

from tests.o2 import test_model_lease_create_hook_package as o2_package


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromName(
        "ModelLeaseCreateHookPackageTests."
        "test_public_facade_unchanged_and_header_tree_pinned",
        module=o2_package,
    )
