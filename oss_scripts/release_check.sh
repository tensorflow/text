#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

# Set up for gbash.sh
sudo ln -s "${KOKORO_GFILE_DIR}/gbash.sh" /usr/bin/gbash.sh
export GBASH_ROOT="${KOKORO_GFILE_DIR}"

ls "${KOKORO_GFILE_DIR}"

source gbash.sh || exit
sudo apt-get install jq
source module check_versions.sh

release_version="$(check_releases)"
echo "Release version = $release_version"
if [ -z $release_version ]; then
  exit 0
fi
exit 1
