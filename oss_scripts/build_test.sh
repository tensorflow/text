#!/bin/bash

PY_VERSION=${1}

bazel version

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# Run configure.
./oss_scripts/configure.sh

# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

ls ${KOKORO_ARTIFACTS_DIR}
deactivate

cd ${KOKORO_ARTIFACTS_DIR}
python3 -m pip install -U auditwheel==1.8.0
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

  cd ${KOKORO_ARTIFACTS_DIR}
  python3 -m pip install -U twine
  twine upload tensorflow_text-*-manylinux1_x86_64.whl
fi
