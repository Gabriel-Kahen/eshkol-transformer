SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build \
	build-ci-core build-ci-contracts build-ci-checkpoint build-ci-parameters \
	build-ci-tokenizer-byte build-ci-tokenizer-bpe \
	build-ci-tokenizer-bpe-boundary build-ci-dataset \
	test test-after-build test-acceptance-predecessors-after-build \
	test-acceptance-c2-after-build test-ci-core-after-build \
	test-ci-optimizer-after-build \
	test-ci-contracts-after-build test-ci-checkpoint-after-build \
	test-ci-parameters-after-build test-ci-tokenizer-byte-after-build \
	test-ci-tokenizer-bpe-after-build \
	test-ci-tokenizer-bpe-boundary-after-build test-ci-dataset-after-build \
	test-ci-topology \
	test-a0 test-a2 test-b0 test-c1 test-c2 test-c2-codec test-c2-core \
	test-c2-checkpoint-inspect test-c2-checkpoint-load test-c2-checkpoint-save \
	test-c2-checkpoint-operational test-c2-public \
	test-c2-d2-cursor-pair test-c2-format \
	test-c2-model-encode test-c2-o2-encode \
	test-c2-persistence-policy test-c2-training-state-owner \
	test-c2-x1-canonical test-d1 test-d2 test-e3-d2 test-e1 test-e1b \
	test-i1 test-i2 test-i2-native test-k1 test-k2 test-l2 test-l3s test-e3-metrics test-n2 test-n3k \
	test-o2 test-tr3-o test-p1 test-p1-native test-python-isolation test-q0 \
	test-reference-formats test-t1 test-t2 test-x1 \
	smoke smoke-after-build benchmark benchmark-after-build clean

toolchain:
	/usr/bin/bash scripts/bootstrap-eshkol.sh

configure:
	/usr/bin/bash scripts/configure.sh

build: configure
	/usr/bin/bash scripts/generate-p1-roots.sh --check
	/usr/bin/bash scripts/build.sh
	/usr/bin/bash scripts/build-a2.sh
	/usr/bin/bash scripts/build-p1-identity.sh
	/usr/bin/bash scripts/build-p1-package.sh
	/usr/bin/bash scripts/build-c1.sh
	/usr/bin/bash scripts/build-t1.sh
	/usr/bin/bash scripts/build-t2.sh
	/usr/bin/bash scripts/build-d2.sh
	/usr/bin/bash scripts/build-c2.sh
	/usr/bin/bash scripts/build-m3t.sh
	/usr/bin/bash scripts/build-m3.sh
	/usr/bin/bash scripts/build-g3n.sh

build-ci-core: configure
	/usr/bin/bash scripts/generate-p1-roots.sh --check
	/usr/bin/bash scripts/compile-smoke.sh "$${BUILD_DIR:-$$(pwd)/build}"
	/usr/bin/bash scripts/build-k1.sh
	/usr/bin/bash scripts/build-l2.sh
	/usr/bin/bash scripts/build-l3s.sh
	/usr/bin/bash scripts/build-e3-metrics.sh
	/usr/bin/bash scripts/build-i1.sh
	/usr/bin/bash scripts/build-a2.sh
	/usr/bin/bash scripts/build-i2.sh
	/usr/bin/bash scripts/build-k2.sh
	/usr/bin/bash scripts/build-n2.sh
	/usr/bin/bash scripts/build-n3k.sh
	/usr/bin/bash scripts/build-t2.sh
	/usr/bin/bash scripts/build-d2.sh
	/usr/bin/bash scripts/build-o2.sh
	/usr/bin/bash scripts/build-p1-package.sh

build-ci-contracts: configure
	/usr/bin/bash scripts/build-x1.sh
	/usr/bin/bash scripts/build-d1.sh

build-ci-checkpoint: configure
	/usr/bin/bash scripts/build-c1.sh
	/usr/bin/bash scripts/build-c2.sh

build-ci-parameters: configure

build-ci-tokenizer-byte: configure
	/usr/bin/bash scripts/build-d2.sh

build-ci-tokenizer-bpe: configure

build-ci-tokenizer-bpe-boundary: configure
	/usr/bin/bash scripts/build-d2.sh

build-ci-dataset: configure
	/usr/bin/bash scripts/build-k1.sh
	/usr/bin/bash scripts/build-i1.sh
	/usr/bin/bash scripts/build-d2.sh

test: build
	$(MAKE) test-after-build

test-after-build:
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-l2.sh
	/usr/bin/bash scripts/test-l3s.sh
	/usr/bin/bash scripts/test-e3-metrics.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-i2.sh
	/usr/bin/bash scripts/test-k2.sh
	/usr/bin/bash scripts/test-n2.sh
	/usr/bin/bash scripts/test-n3k.sh
	/usr/bin/bash scripts/test-o2.sh
	/usr/bin/bash scripts/test-tr3-o.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-p1.sh
	/usr/bin/bash scripts/test-d1.sh
	/usr/bin/bash scripts/test-d2.sh
	/usr/bin/bash scripts/test-e3-d2.sh
	/usr/bin/bash scripts/test-c1.sh
	/usr/bin/bash scripts/test-c2.sh
	/usr/bin/bash scripts/test-t1.sh
	/usr/bin/bash scripts/test-t2.sh --runtime-only
	/usr/bin/bash scripts/test-t2-boundary.sh
	/usr/bin/bash scripts/test-q0.sh
	/usr/bin/bash scripts/test-m3t.sh
	/usr/bin/bash scripts/test-m3.sh
	/usr/bin/bash scripts/test-g3n.sh

test-acceptance-predecessors-after-build:
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-l2.sh
	/usr/bin/bash scripts/test-l3s.sh
	/usr/bin/bash scripts/test-e3-metrics.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-i2.sh
	/usr/bin/bash scripts/test-k2.sh
	/usr/bin/bash scripts/test-n2.sh
	/usr/bin/bash scripts/test-n3k.sh
	/usr/bin/bash scripts/test-o2.sh
	/usr/bin/bash scripts/test-tr3-o.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-p1.sh
	/usr/bin/bash scripts/test-d1.sh
	/usr/bin/bash scripts/test-d2.sh
	/usr/bin/bash scripts/test-e3-d2.sh
	/usr/bin/bash scripts/test-c1.sh
	/usr/bin/bash scripts/test-t1.sh
	/usr/bin/bash scripts/test-t2.sh --runtime-only
	/usr/bin/bash scripts/test-t2-boundary.sh
	/usr/bin/bash scripts/test-q0.sh
	/usr/bin/bash scripts/test-m3t.sh
	/usr/bin/bash scripts/test-m3.sh
	/usr/bin/bash scripts/test-g3n.sh

test-acceptance-c2-after-build:
	/usr/bin/bash scripts/test-c2.sh

test-ci-core-after-build:
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-l2.sh
	/usr/bin/bash scripts/test-l3s.sh
	/usr/bin/bash scripts/test-e3-metrics.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-i2.sh
	/usr/bin/bash scripts/test-k2.sh
	/usr/bin/bash scripts/test-n2.sh
	/usr/bin/bash scripts/test-n3k.sh
	/usr/bin/bash scripts/test-q0.sh

test-ci-optimizer-after-build:
	/usr/bin/bash scripts/test-o2.sh
	/usr/bin/bash scripts/test-tr3-o.sh

test-ci-contracts-after-build:
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-d1.sh

test-ci-checkpoint-after-build:
	/usr/bin/bash scripts/test-c1.sh

.PHONY: test-ci-clean-build-after-build test-ci-c2-format-after-build \
	test-ci-c2-state-after-build test-ci-c2-save-after-build \
	test-ci-c2-load-after-build test-ci-c2-public-after-build \
	test-ci-c2-operational-after-build

test-ci-clean-build-after-build:
	$(MAKE) smoke-after-build benchmark-after-build

test-ci-c2-format-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-format

test-ci-c2-state-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-state

test-ci-c2-save-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-save

test-ci-c2-load-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-load

test-ci-c2-public-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-public

test-ci-c2-operational-after-build:
	/usr/bin/bash scripts/test-c2.sh --group c2-operational

test-ci-parameters-after-build:
	/usr/bin/bash scripts/test-p1.sh

test-ci-tokenizer-byte-after-build:
	/usr/bin/bash scripts/test-t1.sh

test-ci-tokenizer-bpe-after-build:
	/usr/bin/bash scripts/test-t2.sh --runtime-only

test-ci-tokenizer-bpe-boundary-after-build:
	/usr/bin/bash scripts/test-t2-boundary.sh

test-ci-dataset-after-build:
	/usr/bin/bash scripts/test-d2.sh
	/usr/bin/bash scripts/test-e3-d2.sh

test-ci-topology:
	/usr/bin/bash scripts/check-ci-topology.sh

test-a0: build
	/usr/bin/bash scripts/check_a0_api_contract.sh

test-a2: build
	/usr/bin/bash scripts/test-a2.sh

test-b0:
	/usr/bin/bash scripts/test-b0.sh

test-k1: build
	/usr/bin/bash scripts/test-k1.sh

test-l2: build
	/usr/bin/bash scripts/test-l2.sh

test-e1: configure
	/usr/bin/bash scripts/test-e1.sh

test-e1b: configure
	/usr/bin/bash scripts/test-e1b.sh

test-i1: build
	/usr/bin/bash scripts/test-i1.sh

test-i2: build
	/usr/bin/bash scripts/test-i2.sh

test-i2-native: build-ci-core
	/usr/bin/bash scripts/test-i2-native.sh

test-k2: build
	/usr/bin/bash scripts/test-k2.sh

test-n2: build
	/usr/bin/bash scripts/test-n2.sh

test-n3k: build
	/usr/bin/bash scripts/test-n3k.sh

test-o2: build
	/usr/bin/bash scripts/test-o2.sh

# Private TR3 successor seam; no public trainer aggregate is built here.
test-tr3-o:
	/usr/bin/bash scripts/test-tr3-o.sh

test-x1: configure
	/usr/bin/bash scripts/test-x1.sh

test-p1: configure
	/usr/bin/bash scripts/test-p1.sh

test-p1-native: configure
	/usr/bin/bash scripts/test-p1-native.sh

test-d1: build
	/usr/bin/bash scripts/test-d1.sh

test-d2: build
	/usr/bin/bash scripts/test-d2.sh

test-c1: build
	/usr/bin/bash scripts/test-c1.sh

test-c2: configure
	/usr/bin/bash scripts/test-c2.sh

test-c2-format:
	/usr/bin/bash scripts/test-c2-format.sh

test-c2-codec:
	/usr/bin/bash scripts/test-c2-codec.sh

test-c2-core:
	/usr/bin/bash scripts/test-c2-core.sh

test-c2-checkpoint-inspect:
	/usr/bin/bash scripts/test-c2-checkpoint-inspect.sh

test-c2-checkpoint-load:
	/usr/bin/bash scripts/test-c2-checkpoint-load.sh

test-c2-checkpoint-save:
	/usr/bin/bash scripts/test-c2-checkpoint-save.sh

test-c2-checkpoint-operational:
	/usr/bin/bash scripts/test-c2-checkpoint-operational.sh

test-c2-public:
	/usr/bin/bash scripts/test-c2-public.sh

test-c2-persistence-policy:
	/usr/bin/bash scripts/test-c2-persistence-policy.sh

test-c2-x1-canonical:
	/usr/bin/bash scripts/test-c2-x1-canonical.sh

test-c2-d2-cursor-pair:
	/usr/bin/bash scripts/test-c2-d2-cursor-pair.sh

test-c2-training-state-owner:
	/usr/bin/bash scripts/test-c2-training-state-owner.sh

test-c2-model-encode:
	/usr/bin/bash scripts/test-c2-model-encode.sh

test-c2-o2-encode:
	/usr/bin/bash scripts/test-c2-o2-encode.sh

test-t1: build
	/usr/bin/bash scripts/test-t1.sh

test-t2: build
	/usr/bin/bash scripts/test-t2.sh

test-q0: build
	/usr/bin/bash scripts/test-q0.sh

test-python-isolation:
	python3 -m unittest -v tests.q0.test_python_isolation

test-reference-formats:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
		tests.t1.test_reference \
		tests.t2.test_reference

smoke: build
	$(MAKE) smoke-after-build

smoke-after-build:
	/usr/bin/bash scripts/smoke.sh

benchmark: build
	$(MAKE) benchmark-after-build

benchmark-after-build:
	/usr/bin/bash scripts/benchmark.sh

clean:
	/usr/bin/bash scripts/clean.sh

.PHONY: test-ci-m3t-after-build test-m3t
test-ci-m3t-after-build:
	/usr/bin/bash scripts/test-m3t.sh

test-m3t: configure
	/usr/bin/bash scripts/ci-build-prerequisites.sh diagnostic-transport
	/usr/bin/bash scripts/test-m3t.sh

.PHONY: test-ci-m3-after-build test-m3
test-ci-m3-after-build:
	/usr/bin/bash scripts/test-m3.sh

test-m3: configure
	/usr/bin/bash scripts/ci-build-prerequisites.sh model-composition
	/usr/bin/bash scripts/test-m3.sh

# Focused standalone gate; the CI core consumes its once-built prerequisites.
test-l3s: configure
	/usr/bin/bash scripts/build-k1.sh
	/usr/bin/bash scripts/build-l2.sh
	/usr/bin/bash scripts/build-i1.sh
	/usr/bin/bash scripts/build-i2.sh
	/usr/bin/bash scripts/build-l3s.sh
	/usr/bin/bash scripts/test-l3s.sh

.PHONY: test-ci-g3n-after-build test-g3n
test-ci-g3n-after-build:
	/usr/bin/bash scripts/test-g3n.sh

test-g3n: configure
	/usr/bin/bash scripts/ci-build-prerequisites.sh g3n-forward
	/usr/bin/bash scripts/test-g3n.sh

# Source-only shared layer; no production consumer aggregate is built.
.PHONY: test-m3cg
test-m3cg: configure
	/usr/bin/bash scripts/test-m3cg.sh

# Bounded metrics gate shares the canonical numerical prerequisites in CI.
test-e3-metrics: configure
	/usr/bin/bash scripts/build-k1.sh
	/usr/bin/bash scripts/build-l2.sh
	/usr/bin/bash scripts/build-i1.sh
	/usr/bin/bash scripts/build-i2.sh
	/usr/bin/bash scripts/build-l3s.sh
	/usr/bin/bash scripts/build-e3-metrics.sh
	/usr/bin/bash scripts/test-e3-metrics.sh

# Private E3-D2 prerequisite builds its exact test tuple locally.
test-e3-d2: configure
	/usr/bin/bash scripts/test-e3-d2.sh
