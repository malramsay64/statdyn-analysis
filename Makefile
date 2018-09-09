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
	isort --check-only --recursive src/
	black --check src/
	pylint src/
	mypy src/
	pytest

deploy:


.PHONY: help test clean deploy

# vim:ft=make
#
