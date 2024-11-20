#!/usr/bin/env bash

clone_repo() {
    local REPO_URL="${1}"
    local REPO_DIR="${2}"
    if [ ! -d "${REPO_DIR}" ]; then
        git clone "${REPO_URL}" "${REPO_DIR}"
    else
        echo "Repository '${REPO_DIR}' already exists. Skipping."
    fi
}

# compass-docker
clone_repo "git@github.com:voges/compass-docker.git" "external/compass-docker"

# gears-fork
clone_repo "git@github.com:voges/gears-fork.git" "external/gears-fork"

# metavis
clone_repo "git@github.com:GuilleDufortFing/metavis.git" "external/metavis"

# pertdata
clone_repo "git@github.com:voges/pertdata.git" "external/pertdata"
