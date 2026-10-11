/* Runner I candidate and upstream/H history pins; no I review or release authority. */
struct pin { const char *path; const char *sha256; };
static const struct pin PINS[] = {
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-runner-20261011-i/SHA256SUMS-RUNNER", "f59779e33af754ca053872a617e870b69f02bf000197db0e63867b0eb01ca208"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-runner-independent-review-20261011-h/SHA256SUMS-REVIEW", "9ff3930da396010cb23205b4ba2a0f2e733dc8b20eb8c89a1a6d751146d59e16"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-runner-root-source-review-20261011-h/SHA256SUMS-ROOT-REVIEW", "e36c69f740a2c0d6871e0c7b882da5bf6488690ca9785ba867fd773c72c521ca"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-build-plan-20261011-fb/SHA256SUMS-PLAN", "66b4d8e5ab1410d6bef533da88cfe998a4a97eb847e24af0a67e5052aebda4e2"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-build-plan-20261011-gb/SHA256SUMS-PLAN", "2806213768bfdae0a4fa8a09b46ef7bc5a74c2d315496cec7f8c2f5cd6fd5d28"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-config-run-20261011-fb/SHA256SUMS-RUN", "9fe938db7abb7858d4de7c7b91953a47858be110d6d8a953aa5c1ef74ab18260"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-config-run-20261011-gb/SHA256SUMS-RUN", "f5acfceb9910dea3546149c383610d36fd6762e6464e3c8c6bf72643d41a898b"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-graph-audit-20261011-a/SHA256SUMS-AUDIT", "340c92bbd7540e6ddf6c977f5fddbec3d656f10833a4f8cec95c0b41c4500c98"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-parser-20261011-d/SHA256SUMS-PARSER", "c89a991654ea852b67301b6feeacb9702bd4d570f38a3a35220ecc4e7804e970"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-parser-independent-review-20261011-d/SHA256SUMS-REVIEW", "880f2c369973ae586757cdb45341e99357d24f830db6228eb2c51126b36039b6"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-inventory-parser-root-review-20261011-d/SHA256SUMS-ROOT-REVIEW", "2631dcb9b9c3e91bfecb457b0d3d8b867fb5848fc8ba05d4a3da5236867c77a7"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-independent-plan-review-20261011-b/SHA256SUMS-REVIEW", "d8d9406816e24814ba0dc0d8fa2ad148a19be0f8929516b616a0e202ea133b1c"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-plan-20261011-b/SHA256SUMS-PLAN", "5eed63c2b427d52ad07c4e1e3a3f86f38c1ba077b097860f1892f0f68da96afd"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-root-plan-review-20261011-b/SHA256SUMS-ROOT-REVIEW", "5116e363dba86ca29eb87ec763985f5f0803a2ae80c90e910ad5c492606a0b7c"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-tool-closure-20261011-d/SHA256SUMS-CLOSURE", "b128967e98ba884eba81d337b198dbc731c90a623847fb2b85598c1f2a37d344"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-tool-closure-review-20261011-d/SHA256SUMS-REVIEW", "ac67b821156a1ec24e8d275f6dc6fad2cde966f7b1b667d289e3ea97d0346f0a"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-parity-tool-closure-root-review-20261011-d/SHA256SUMS-ROOT-REVIEW", "66cb647a1b238c840ac4080e4613f859f8b156c40e60956032b0b7e332c67e84"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-build-run-20261011-fb/SHA256SUMS-RUN", "ffbe5393a90f12134fb64693511e9d99ed50e38940032e61fec7d777c58a81d3"},
    {"/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-build-run-20261011-gb/SHA256SUMS-RUN", "a02d8a0897904311641e4fc7f7f3eb32c6bc6d1fce39c7edf852cb2df92d7e34"},
};
static const unsigned PIN_COUNT = 19;
