FROM quay.io/pypa/manylinux1_x86_64

ENV PATH="/opt/python/cp36-cp36m/bin:${PATH}"

RUN pip install pipenv

RUN git clone https://github.com/malramsay64/statdyn-analysis.git

WORKDIR statdyn-analysis

RUN pipenv install --dev --three
RUN pipenv run pip install .

ENV SHELL=/bin/bash

ENTRYPOINT ["pipenv", "run"]
