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
	pip3 install -U pip
	pip3 install pipenv
	pipenv install --dev --three --ignore-pipfile
	pipenv run pip install -e .
	pipenv run pip install codecov

test:
	pipenv run pytest
	pipenv run codecov


deploy:
	pipenv run python setup.py sdist
	pipenv run twine upload --skip-existing dist/*

clean:
	rm -f dist/*

.PHONY: help test clean deploy

# vim:ft=make
#
