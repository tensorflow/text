#!/usr/bin/env bash
# Tool to build the TensorFlow Text pip package.
#
# Usage:
#   bazel build oss_scripts/pip_package:build_pip_package
#   bazel-bin/oss_scripts/build_pip_package
#
# Arguments:
#   output_dir: An output directory. Defaults to `/tmp/tensorflow_text_pkg`.

set -e  # fail and exit on any command erroring

die() {
  echo >&2 "$@"
  exit 1
}

osname="$(uname -s)"
echo $osname
readlinkcmd=readlink
if [[ $osname == "Darwin" ]]; then
  readlinkcmd=greadlink
fi

main() {
  local output_dir="$1"

  if [[ -z "${output_dir}" ]]; then
    output_dir="/tmp/tensorflow_text_pkg"
  fi
  mkdir -p ${output_dir}
  output_dir=$($readlinkcmd -f "${output_dir}")
  echo "=== Destination directory: ${output_dir}"

  if [[ ! -d "bazel-bin/tensorflow_text" ]]; then
    die "Could not find bazel-bin. Did you run from the root of the build tree?"
  fi

  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  echo "=== Using tmpdir ${temp_dir}"

  local runfiles="bazel-bin/oss_scripts/pip_package/build_pip_package.runfiles"
  cp -LR \
      "${runfiles}/org_tensorflow_text/tensorflow_text" \
      "${temp_dir}"
  cp "${runfiles}/org_tensorflow_text/oss_scripts/pip_package/setup.py" \
      "${temp_dir}"
  cp "${runfiles}/org_tensorflow_text/oss_scripts/pip_package/MANIFEST.in" \
      "${temp_dir}"
  cp "${runfiles}/org_tensorflow_text/oss_scripts/pip_package/LICENSE" \
      "${temp_dir}"

  pushd "${temp_dir}" > /dev/null

  # Build pip package
  python setup.py bdist_wheel --universal
  cp dist/*.whl "${output_dir}"
}

main "$@"
