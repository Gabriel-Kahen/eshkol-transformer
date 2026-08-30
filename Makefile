SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build test test-a0 test-b0 test-e1 test-e1b test-i1 test-k1 test-p1 test-x1 smoke benchmark clean

toolchain:
	/usr/bin/bash scripts/bootstrap-eshkol.sh

configure:
	/usr/bin/bash scripts/configure.sh

build: configure
	/usr/bin/bash scripts/generate-p1-roots.sh --check
	/usr/bin/bash scripts/build.sh
	/usr/bin/bash scripts/build-p1-identity.sh

test: build
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh
	/usr/bin/bash scripts/test-e1.sh
	/usr/bin/bash scripts/test-e1b.sh
	/usr/bin/bash scripts/test-i1.sh
	/usr/bin/bash scripts/test-x1.sh
	/usr/bin/bash scripts/test-p1.sh

test-a0: configure
	/usr/bin/bash scripts/check_a0_api_contract.sh

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

smoke: build
	/usr/bin/bash scripts/smoke.sh

benchmark: build
	/usr/bin/bash scripts/benchmark.sh

clean:
	/usr/bin/bash scripts/clean.sh
