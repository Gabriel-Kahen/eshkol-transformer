SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build test test-a0 test-b0 test-k1 smoke benchmark clean

toolchain:
	/usr/bin/bash scripts/bootstrap-eshkol.sh

configure:
	/usr/bin/bash scripts/configure.sh

build: configure
	/usr/bin/bash scripts/build.sh

test: configure
	/usr/bin/bash scripts/test.sh
	/usr/bin/bash scripts/check_a0_api_contract.sh
	/usr/bin/bash scripts/test-k1.sh

test-a0: configure
	/usr/bin/bash scripts/check_a0_api_contract.sh

test-b0:
	/usr/bin/bash scripts/test-b0.sh

test-k1: configure
	/usr/bin/bash scripts/test-k1.sh

smoke: build
	/usr/bin/bash scripts/smoke.sh

benchmark: build
	/usr/bin/bash scripts/benchmark.sh

clean:
	/usr/bin/bash scripts/clean.sh
