#!/bin/bash
set -e  # fail and exit on any command erroring

osname="$(uname -s | tr 'A-Z' 'a-z')"

if [[ $osname == "darwin" ]]; then
  # Update to macos extensions
  sed -i '' 's/".so"/".dylib"/' tensorflow_text/tftext.bzl
  perl -pi -e "s/(load_library.load_op_library.*)\\.so'/\$1.dylib'/" $(find tensorflow_text/python -type f)
  export CC_OPT_FLAGS='-mavx'
fi

# Run configure.
source oss_scripts/configure.sh

# Set tensorflow version
if [[ $osname != "darwin" ]] || [[ ! $(sysctl -n machdep.cpu.brand_string) =~ "Apple" ]]; then
  source oss_scripts/prepare_tf_dep.sh
fi

# Build the pip package.
bazel run --enable_runfiles //oss_scripts/pip_package:build_pip_package -- "$(realpath .)"
