#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands change

PY_VERSION=${1}

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout the release branch if specified.
git checkout -f "${RELEASE_BRANCH:-master}"

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate
bazel version

# Run configure.
./oss_scripts/configure.sh

# Build the pip package
bazel build --config=manylinux2010 oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

ls ${KOKORO_ARTIFACTS_DIR}
deactivate

cd ${KOKORO_ARTIFACTS_DIR}
sudo python3 -m pip install -U auditwheel==1.8.0
python3 -m pip install -U wheel==0.31.1
for f in *.whl; do auditwheel repair -w . --plat manylinux1_x86_64 $f; done

# Release
if [[ "$UPLOAD_TO_PYPI" == "upload" ]]; then
  PYPI_PASSWD="$(cat "$KOKORO_KEYSTORE_DIR"/74641_tftext_pypi_automation_passwd)"
  cat >~/.pypirc <<EOL
[pypi]
username = __token__
password = ${PYPI_PASSWD}
EOL

  # create virtual env
  python3 -m virtualenv /tmp/env
  source /tmp/env/bin/activate
  pip install twine

  cd ${KOKORO_ARTIFACTS_DIR}
  twine upload tensorflow_text-*-manylinux1_x86_64.whl
fi
