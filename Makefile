SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build test test-b0 smoke benchmark clean

toolchain:
	/usr/bin/bash scripts/bootstrap-eshkol.sh

configure:
	/usr/bin/bash scripts/configure.sh

build: configure
	/usr/bin/bash scripts/build.sh

test: configure
	/usr/bin/bash scripts/test.sh

test-b0:
	/usr/bin/bash scripts/test-b0.sh

smoke: build
	/usr/bin/bash scripts/smoke.sh

benchmark: build
	/usr/bin/bash scripts/benchmark.sh

clean:
	/usr/bin/bash scripts/clean.sh
