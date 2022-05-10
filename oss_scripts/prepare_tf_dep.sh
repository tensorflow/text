#!/bin/bash
set -e  # fail and exit on any command erroring

if (which python) | grep -q "python"; then
  installed_python="python"
elif (which python3) | grep -q "python3"; then
  installed_python="python3"
fi

ext=""
osname="$(uname -s | tr 'A-Z' 'a-z')"
if [[ "${osname}" == "darwin" ]]; then
  ext='""'
fi

# update setup.nightly.py with tf version
tf_version=$($installed_python -c 'import tensorflow as tf; print(tf.__version__)')
echo "$tf_version"
sed -i $ext "s/project_version = 'REPLACE_ME'/project_version = '${tf_version}'/" oss_scripts/pip_package/setup.nightly.py
# update __version__
sed -i $ext "s/__version__ = .*\$/__version__ = \"${tf_version}\"/" tensorflow_text/__init__.py

# Get commit sha of installed tensorflow
# For some unknown reason this now needs to be split into two commands on Windows
short_commit_sha=$($installed_python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1)
if [[ "${osname}" == "darwin" ]]; then
  short_commit_sha=$(echo $short_commit_sha | perl -nle 'print $& while m{(?<=-g)[0-9a-f]*$}g')
else
  short_commit_sha=$(echo $short_commit_sha | grep -oP '(?<=-g)[0-9a-f]*$')
fi
commit_sha=$(curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')

# Update TF dependency to installed tensorflow
sed -E -i $ext "s/strip_prefix = \"tensorflow-2.+\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
sed -E -i $ext "s|\"https://github.com/tensorflow/tensorflow/archive/v.+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i $ext "s/sha256 = \"${prev_shasum}\",//" WORKSPACE
