SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build test test-a0 test-a2 test-b0 test-c1 test-d1 test-e1 test-e1b test-i1 test-k1 test-p1 test-python-isolation test-t1 test-x1 smoke benchmark clean

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

test: build
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-a2.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-p1.sh
	/usr/bin/bash scripts/test-d1.sh
	/usr/bin/bash scripts/test-c1.sh
	/usr/bin/bash scripts/test-t1.sh
	python3 -m unittest -v tests.q0.test_python_isolation

test-a0: build
	/usr/bin/bash scripts/check_a0_api_contract.sh

test-a2: build
	/usr/bin/bash scripts/test-a2.sh

test-b0:
	/usr/bin/bash scripts/test-b0.sh

test-k1: build
	/usr/bin/bash scripts/test-k1.sh

test-e1: configure
	/usr/bin/bash scripts/test-e1.sh

test-e1b: configure
	/usr/bin/bash scripts/test-e1b.sh

test-i1: build
	/usr/bin/bash scripts/test-i1.sh

test-x1: configure
	/usr/bin/bash scripts/test-x1.sh

test-p1: configure
	/usr/bin/bash scripts/test-p1.sh

test-d1: build
	/usr/bin/bash scripts/test-d1.sh

test-c1: build
	/usr/bin/bash scripts/test-c1.sh

test-t1: build
	/usr/bin/bash scripts/test-t1.sh

test-python-isolation:
	python3 -m unittest -v tests.q0.test_python_isolation

smoke: build
	/usr/bin/bash scripts/smoke.sh

benchmark: build
	/usr/bin/bash scripts/benchmark.sh

clean:
	/usr/bin/bash scripts/clean.sh
