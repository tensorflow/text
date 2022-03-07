#!/bin/bash
set -e  # fail and exit on any command erroring

if (which python) | grep -q "python"; then
  installed_python="python"
elif (which python3) | grep -q "python3"; then
  installed_python="python3"
fi

# update setup.nightly.py with tf version
tf_version=$($installed_python -c 'import tensorflow as tf; print(tf.__version__)')
echo "$tf_version"
sed -i "s/project_version = 'REPLACE_ME'/project_version = '${tf_version}'/" oss_scripts/pip_package/setup.nightly.py
# update __version__
sed -i "s/__version__ = .*\$/__version__ = \"${tf_version}\"/" tensorflow_text/__init__.py

# Get commit sha of installed tensorflow
short_commit_sha=$($installed_python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1 | grep -oP '(?<=-g)[0-9a-f]*$')
commit_sha=$(curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')

# Update TF dependency to installed tensorflow
sed -i "s/strip_prefix = \"tensorflow-.\+\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
sed -i "s|\"https://github.com/tensorflow/tensorflow/archive/.\+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE
