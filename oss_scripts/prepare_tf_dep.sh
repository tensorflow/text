#!/bin/bash
set -e  # fail and exit on any command erroring

ext=""
osname="$(uname -s | tr 'A-Z' 'a-z')"
if [[ "${osname}" == "darwin" ]]; then
  ext='""'
fi

# Update the setup file with current tf version. build_pip_package uses
# setup.nightly.py for nightly builds and setup.py otherwise.
setup_file="oss_scripts/pip_package/setup.py"
if [[ "$IS_NIGHTLY" == "nightly" ]]; then
  setup_file="oss_scripts/pip_package/setup.nightly.py"
fi
tf_version=$(bazel run //oss_scripts/pip_package:tensorflow_build_info -- version)
echo "Updating $setup_file to version $tf_version"
sed -i $ext "s/project_version = '.*'/project_version = '${tf_version}'/" "$setup_file"
# Update __version__.
echo "Updating __init__.py to version $tf_version"
sed -i $ext "s/__version__ = .*\$/__version__ = \"${tf_version}\"/" tensorflow_text/__init__.py

# Get git commit sha of installed tensorflow.
echo "Querying commit SHA"
short_commit_sha=$(bazel run  //oss_scripts/pip_package:tensorflow_build_info -- git_version)
if [[ "$short_commit_sha" == "unknown" ]]; then
  # Some nightly builds report "unknown" for tf.__git_version.
  echo 'TF git version "unknown", assuming nightly.'
  commit_slug='nightly'
else
  if [[ "${osname}" == "darwin" ]]; then
    short_commit_sha=$(echo $short_commit_sha | perl -nle 'print $& while m{(?<=-g)[0-9a-f]*$}g')
  else
    short_commit_sha=$(echo $short_commit_sha | grep -oP '(?<=-g)[0-9a-f]*$')
  fi
  echo "Found tensorflow commit sha: $short_commit_sha"
  commit_slug=$(curl -s "https://api.github.com/repos/tensorflow/tensorflow/commits/$short_commit_sha" | grep "sha" | head -n 1 | cut -d '"' -f 4)
fi

# Update the TensorFlow and XLA modules in MODULE.bazel to the installed
# tensorflow. Both modules come from the same TensorFlow archive.
echo "Updating MODULE.bazel to use TensorFlow commit $commit_slug"
tf_integrity=$(awk '/module_name = "tensorflow"/ {found = 1} found && /integrity = / {print; exit}' MODULE.bazel | cut -d'"' -f2)
sed -E -i $ext "s|strip_prefix = \"tensorflow-[^\"/]+(/third_party/xla)?\",|strip_prefix = \"tensorflow-${commit_slug}\\1\",|" MODULE.bazel
sed -E -i $ext "s|\"https://github.com/tensorflow/tensorflow/archive/[^\"]+\.tar\.gz\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_slug}.tar.gz\"|" MODULE.bazel
# The integrity of the previous archive does not match the new archive.
if [[ -n "$tf_integrity" ]]; then
  sed -i $ext "\\|integrity = \"${tf_integrity}\",|d" MODULE.bazel
fi
