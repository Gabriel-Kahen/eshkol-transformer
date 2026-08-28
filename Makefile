SHELL := /usr/bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: toolchain configure build test smoke clean

toolchain:
	/usr/bin/bash scripts/bootstrap-eshkol.sh

configure:
	/usr/bin/bash scripts/configure.sh

build: configure
	/usr/bin/bash scripts/build.sh

test: configure
	/usr/bin/bash scripts/test.sh

smoke: build
	/usr/bin/bash scripts/smoke.sh

clean:
	/usr/bin/bash scripts/clean.sh
