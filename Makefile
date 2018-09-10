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
	pre-commit install-hooks

test:
	python3 -m isort --check-only --recursive src/
	python3 -m black --check src/
	python3 -m pylint src/
	python3 -m mypy src/
	python3 -m pytest

lock:
	docker run -it\
		-v $(shell pwd)/environment.yml:/srv/environment.yml:Z \
		continuumio/miniconda3:4.5.4 \
		conda env create -f /srv/environment.yml && \
		conda activate sdanalysis-dev && \
		conda env export > environment.lock

deploy:


.PHONY: help test clean deploy

# vim:ft=make
#
