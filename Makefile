#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup:
	conda env update

test:
	python3 -m isort --check-only --recursive src/
	python3 -m black --check src/
	python3 -m pylint src/
	python3 -m mypy src/
	python3 -m pytest

deploy:


.PHONY: help test clean deploy

# vim:ft=make
#
