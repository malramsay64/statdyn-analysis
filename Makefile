#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#

PREFIX := "$(HOME)/miniconda/bin"
SHELL := /bin/bash
miniconda := download/miniconda.sh

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup: $(PREFIX)
	$(PREFIX)/conda config --set always_yes yes --set changeps1 no
	$(PREFIX)/conda update -q conda
	$(PREFIX)/conda install conda-env
	$(PREFIX)/conda info -a
	$(PREFIX)/conda env update
	$(PREFIX)/pip install codecov
	source $(PREFIX)/activate sdanalysis-dev; \
	python setup.py install --single-version-externally-managed --record record.txt

test:
	source $(PREFIX)/activate sdanalysis-dev; pytest
	$(PREFIX)/codecov

deploy: pre-deploy
	@echo "Deploying to PyPI..."
	source $(PREFIX)/activate sdanalysis-dev; python setup.py bdist
	$(PREFIX)/twine upload dist/*.tar.gz
	@echo "Deploying to Anaconda..."
	$(PREFIX)/conda build .

pre-deploy:
	$(PREFIX)/conda install -n root anaconda-client conda-build
	$(PREFIX)/conda install -n root twine
	$(PREFIX)/conda config --set anaconda_upload yes

$(PREFIX): $(miniconda)
	bash $< -b -u -p $(shell dirname "$@")

$(miniconda):
	mkdir -p $(shell dirname "$@")
ifeq ($(uname -s), "Darwin")
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $@
else
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $@
endif

.PHONY: help test

# vim:ft=make
#
