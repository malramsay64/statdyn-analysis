#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#
ifeq ($(shell uname -s),Darwin)
CMD:= pipenv run
else
CMD:= docker run -e TWINE_USERNAME=${TWINE_USERNAME} -e TWINE_PASSWORD=${TWINE_PASSWORD} build_wheel
endif

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup:
	echo $(CMD)
ifeq ($(shell uname -s),Darwin)
	pip3 install -U pip
	pip3 install pipenv
	pipenv install --dev --three
else
	docker build -t build_wheel .
endif

test:
	$(CMD) pytest

deploy:
ifeq ($(shell uname -s),Darwin)
	pipenv run python setup.py bdist_wheel
	pipenv run twine upload --skip-existing dist/*
else
	$(CMD) bash -c "python setup.py bdist_wheel && \
		auditwheel repair dist/*.whl && \
		twine upload --skip-existing wheelhouse/*"
endif

clean:
	rm -f dist/*

.PHONY: help test clean deploy

# vim:ft=make
#
