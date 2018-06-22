FROM quay.io/pypa/manylinux1_x86_64

ENV PATH="/opt/python/cp36-cp36m/bin:${PATH}"

RUN pip3 install -U pip
RUN pip3 install pipenv

COPY . statdyn-analysis

WORKDIR statdyn-analysis

RUN pipenv install --dev --three
RUN pipenv run pip install codecov

ENV SHELL=/bin/bash

ENTRYPOINT ["pipenv", "run"]
