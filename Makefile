SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build build-fast-core build-fast-contracts \
	build-fast-checkpoint build-fast-state \
	test test-after-build test-fast-core-after-build \
	test-fast-contracts-after-build test-fast-checkpoint-after-build \
	test-fast-state-after-build \
	test-a0 test-a2 test-b0 test-c1 test-d1 test-e1 test-e1b test-i1 \
	test-i2 test-i2-native test-k1 test-l2 test-p1 test-p1-native \
	test-python-isolation test-reference-formats test-t1 test-t2 test-x1 \
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

build-fast-core: configure
	/usr/bin/bash scripts/generate-p1-roots.sh --check
	/usr/bin/bash scripts/compile-smoke.sh "$${BUILD_DIR:-$$(pwd)/build}"
	/usr/bin/bash scripts/build-k1.sh
	/usr/bin/bash scripts/build-l2.sh
	/usr/bin/bash scripts/build-i1.sh
	/usr/bin/bash scripts/build-a2.sh
	/usr/bin/bash scripts/build-i2.sh
	/usr/bin/bash scripts/build-p1-package.sh

build-fast-contracts: configure
	/usr/bin/bash scripts/build-d1.sh

build-fast-checkpoint: configure
	/usr/bin/bash scripts/build-c1.sh

build-fast-state: configure
	/usr/bin/bash scripts/build-t1.sh
	/usr/bin/bash scripts/build-t2.sh

test: build
	$(MAKE) test-after-build

test-after-build:
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-l2.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-i2.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-p1.sh
	/usr/bin/bash scripts/test-d1.sh
	/usr/bin/bash scripts/test-c1.sh
	/usr/bin/bash scripts/test-t1.sh
	/usr/bin/bash scripts/test-t2.sh
	python3 -m unittest -v tests.q0.test_python_isolation

test-fast-core-after-build:
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-l2.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-i2-native.sh

test-fast-contracts-after-build:
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-d1.sh

test-fast-checkpoint-after-build:
	/usr/bin/bash scripts/test-c1.sh

test-fast-state-after-build:
	/usr/bin/bash scripts/test-p1-native.sh
	$(MAKE) test-reference-formats
	$(MAKE) test-python-isolation

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

test-i2-native: build-fast-core
	/usr/bin/bash scripts/test-i2-native.sh

test-x1: configure
	/usr/bin/bash scripts/test-x1.sh

test-p1: configure
	/usr/bin/bash scripts/test-p1.sh

test-p1-native: configure
	/usr/bin/bash scripts/test-p1-native.sh

test-d1: build
	/usr/bin/bash scripts/test-d1.sh

test-c1: build
	/usr/bin/bash scripts/test-c1.sh

test-t1: build
	/usr/bin/bash scripts/test-t1.sh

test-t2: build
	/usr/bin/bash scripts/test-t2.sh

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
