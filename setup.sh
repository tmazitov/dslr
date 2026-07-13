#!/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install --index-url https://pypi.org/simple matplotlib
pip install --index-url https://pypi.org/simple pandas
