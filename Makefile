#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#


help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup:
	pip3 install pipenv
	pipenv install --dev --three
	pipenv run -- pip install .

test:
	pipenv run pytest

deploy: clean
ifeq ($(shell uname), 'Darwin')
	pipenv run python setup.py bdist_wheel
else
	pipenv run python setup.py sdist
endif
	pipenv run twine upload --skip-existing dist/*.tar.gz

clean:
	rm -f dist/*

.PHONY: help test clean deploy

# vim:ft=make
#
