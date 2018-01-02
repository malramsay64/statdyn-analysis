#! /bin/sh
#
# deploy_pypi.sh
# Copyright (C) 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#

cat << EOF > ~/.pypirc
[distutils]
index-servers = pypi

[pypi]
username=malramsay64
EOF

pip install -y twine

twine upload dist/*
