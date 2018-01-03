#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
export PATH :="$(HOME)/miniconda/bin:$(PATH)"

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup: install_miniconda
	bash miniconda.sh -b -u -p "$(HOME)/miniconda"
	@echo $(PATH)
	hash -r
	conda config --set always_yes yes --set changeps1 no
	conda update -q conda
	conda install conda-env
	conda info -a
	conda env update
	pip install codecov
	source activate sdanalysis-dev; \
	python setup.py install --single-version-externally-managed --record record.txt

test:
	source activate sdanalysis-dev; pytest
	codecov

deploy: pre-deploy
	@echo "Deploying to PyPI..."
	source activate sdanalysis-dev; python setup.py bdist
	twine upload dist/*.tar.gz
	@echo "Deploying to Anaconda..."
	conda build .

pre-deploy:
	conda install -n root anaconda-client conda-build
	conda install -n root twine
	conda config --set anaconda_upload yes

install_miniconda:
ifeq ($(uname -s), "Darwin")
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
else
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
endif

.PHONY: help test

# vim:ft=make
#
