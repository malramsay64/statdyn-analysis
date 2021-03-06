#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#
lockfile = environment-lock.txt

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"

setup:
	conda env create
	pre-commit install-hooks

lint:
	python3 -m isort --check-only --recursive src/
	python3 -m isort --check-only --recursive test/
	python3 -m black --check src/
	python3 -m black --check test/
	python3 -m flake8 src/
	python3 -m flake8 test/
	python3 -m mypy src/

test:
	python3 -m pytest

test-docs:
	$(MAKE) -C docs html SPHINXOPTS="-W"

install:
	pip install -e . --no-deps

docs:
	$(MAKE) -C docs html


.PHONY: help test clean deploy docs

# vim:ft=make
