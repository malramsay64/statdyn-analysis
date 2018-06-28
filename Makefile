#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#
ifeq ($(shell uname -s),Darwin)
	USE_DOCKER:=
else
	USE_DOCKER:=true
endif

ifdef USE_DOCKER
# Run docker container 'build_wheel' passing environment variables to it
CMD:= docker run -e TWINE_USERNAME=${TWINE_USERNAME} -e TWINE_PASSWORD=${TWINE_PASSWORD} build_wheel
else
CMD:= pipenv run
endif

help:
	@echo "Usage:"
	@echo "    make help       show this message"
	@echo "    make setup      install dependencies and create environment"
	@echo "    make test       run the test suite"
	@echo "    make deploy     deploy application"

setup:
	echo $(CMD)
ifdef USE_DOCKER
	docker build -t build_wheel .
else
	pip3 install -U pip
	pip3 install pipenv
	pipenv install --dev --three
	pipenv run pip install -e .
	pipenv run pip install codecov
endif

test:
ifdef USE_DOCKER
	$(CMD) bash -c "pytest && codecov"
else
	pipenv run pytest
	pipenv run codecov
endif


deploy:
ifdef USE_DOCKER
	$(CMD) bash -c "python setup.py bdist_wheel && \
		auditwheel repair dist/*.whl && \
		twine upload --skip-existing wheelhouse/*"
else
	pipenv run python setup.py bdist_wheel
	pipenv run twine upload --skip-existing dist/*
endif

clean:
	rm -f dist/*

.PHONY: help test clean deploy

# vim:ft=make
#
