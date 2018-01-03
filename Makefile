#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#

PREFIX := $(HOME)/miniconda/bin
SHELL := /bin/bash
miniconda := download/miniconda.sh

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup: $(PREFIX)
	( \
		export PATH=$(PREFIX):$(PATH); \
		conda config --set always_yes yes --set changeps1 no; \
		conda update -q conda ; \
		conda install conda-env; \
		conda info -a; \
		conda env update; \
		source activate sdanalysis-dev; \
		pip install codecov; \
		python setup.py install --single-version-externally-managed --record record.txt; \
	)

test:
	source $(PREFIX)/activate sdanalysis-dev; pytest; codecov

deploy: pre-deploy
	@echo "Deploying to PyPI..."
	( \
		export PATH=$(PREFIX):$(PATH); \
		source activate sdanalysis-dev; \
		python setup.py bdist upload; \
	)
	@echo "Deploying to Anaconda..."
	$(PREFIX)/conda build . -c conda-forge -c moble --user $(CONDA_USER) --token $(CONDA_UPLOAD_TOKEN)

pre-deploy:
	( \
		export PATH=$(PREFIX):$(PATH); \
		conda install -n root anaconda-client conda-build; \
		conda config --set anaconda_upload yes; \
	)

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
