#!/usr/bin/env bash

# Remove all editor settings for Python from the VS Code remote settings.
sed --in-place '/"\[python\]": {/,/}/d' /home/vscode/.vscode-server/data/Machine/settings.json

# Install additional packages.
sudo apt-get update && sudo apt-get install --yes shellcheck

# Install metavis dependencies.
sudo apt-get update && sudo apt-get install --yes graphviz libgraphviz-dev
(cd "external/metavis" && pip3 --disable-pip-version-check install --requirement requirements.txt)

# Install pertdata dependencies.
(cd "external/pertdata" && bash install.sh)

# Install the Python dependencies.
pip3 --disable-pip-version-check install --requirement requirements.txt
