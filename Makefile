# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON ?= python
BUILD_DIR := $(shell python3 -c "import sysconfig, sys; plat_name=sysconfig.get_platform(); python_version=sysconfig.get_python_version(); print(f'build/cmake.{plat_name}-{sys.implementation.name}-{python_version}')")
PYTEST := $(PYTHON) -m pytest
NUM_PROCS ?= 8

# Incremental builds

.PHONY: all
all:
	ninja -C $(BUILD_DIR)


# Testing

.PHONY: release-test-unit
release-test-unit:
	cd ascend/examples/pytest_ut && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=load

.PHONY: release-test-inductor
release-test-inductor:
	cd ascend/examples/inductor_cases && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=load

.PHONY: release-test-gen
release-test-gen:
	cd ascend/examples/generalization_cases && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=load

.PHONY: test-unit
test-unit: all release-test-unit

.PHONY: test-inductor
test-inductor: all release-test-inductor

.PHONY: test-gen
test-gen: all release-test-gen


# pip install-ing

.PHONY: dev-install
dev-install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements_dev.txt


