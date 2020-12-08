#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout the release branch if specified.
git checkout -f 2.4

# Export PATH for Python & Bazel
export PATH=$PATH:/c/tools:/c/Python${PY_VERSION//.}:/c/Python${PY_VERSION//.}/Scripts
python -V
bazel version

# Run configure.
bash oss_scripts/configure.sh

# Build the pip package
bazel build --enable_runfiles oss_scripts/pip_package:build_pip_package
bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

ls ${KOKORO_ARTIFACTS_DIR}

# Release
if [[ "$UPLOAD_TO_PYPI" == "upload" ]]; then
  set +x  # Do not log passwords
  PYPI_PASSWD="$(cat "$KOKORO_KEYSTORE_DIR"/74641_tftext_pypi_automation_passwd)"
  cat >~/.pypirc <<EOL
[pypi]
username = __token__
password = ${PYPI_PASSWD}
EOL
  set -x

  # Running Twine on Py3.8 has problems, so we will default to Py3.7
  export PATH=/c/Python37:/c/Python37/Scripts:$PATH
  python -V

  cd ${KOKORO_ARTIFACTS_DIR}
  pip install twine
  twine upload *.whl
fi
