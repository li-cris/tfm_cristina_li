#!/usr/bin/env bash

# Install Cplex
chmod u+x cplex_studio2211.linux_x86_64.bin
chmod u+x install_cplex.expect
./install_cplex.expect

# Make Python virtual environment and activate it
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install numpy # Required by Compass

# Install Cplex Python API
python3 /opt/ibm/ILOG/CPLEX_Studio2211/python/setup.py install

# Install Compass
python3 -m pip install git+https://github.com/yoseflab/Compass.git@7664cb08466510889f9aafb3271140918ac2bf7e
