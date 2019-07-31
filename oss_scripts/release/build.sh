#!/usr/bin/env bash

# DO NOT use this script manually! Called by docker.

set -ex

# Print usage and fail.
function usage() {
  echo "Usage: sudo docker build.sh TF_TEXT_VERSION TF_VERSION" >&2
  exit 1   # Causes caller to exit because we use -e.
}

# Validate arguments.
if [ $# -lt 2 ]; then
  usage
  exit 1
fi
TF_TEXT_VERSION=$1
TF_VERSION=$2  # eg. '2.0.0-beta1'
UPLOAD_TO_PYPI=$3  # 'upload' iff we should upload wheels to pypi

# Initial setup
cd
mkdir wheels
yum install -y java-1.8.0-openjdk-devel wget which findutils binutils gcc tar gzip zip unzip java java-devel git clang zlib-devel gcc-c++
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtools-2.repo
yum install -y devtoolset-2-gcc devtoolset-2-gcc-c++ devtoolset-2-binutils
JAVA_HOME=/usr/lib/jvm/java-1.8.0
CC=/opt/rh/devtoolset-2/root/usr/bin/gcc
if [ ! -f '/usr/local/bin/bazel' ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip
  unzip bazel-0.24.1-dist.zip
  env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
  cp output/bazel /usr/local/bin
fi
PYTHON_VERSIONS=('cp27-cp27mu' 'cp34-cp34m' 'cp35-cp35m' 'cp36-cp36m' 'cp37-cp37m')

# Create wheel files for each python version
for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"
do
  PYTHON_BIN=/opt/python/${PYTHON_VERSION}/bin/python
  $PYTHON_BIN -m pip install -U virtualenv
  # Init python version and download tf code
  cd
  mkdir -p $PYTHON_VERSION
  cd $PYTHON_VERSION
  $PYTHON_BIN -m virtualenv env
  source env/bin/activate
  pip install -U pip six numpy wheel setuptools mock
  if [[ $PYTHON_VERSION == 'cp27-cp27mu' ]]; then
    pip install -U future enum34
  fi
  pip install -U keras_applications==1.0.6 --no-deps
  pip install -U keras_preprocessing==1.0.5 --no-deps
  curl -SsL -O https://github.com/tensorflow/tensorflow/archive/v${TF_VERSION}.zip
  unzip "v${TF_VERSION}.zip"
  mv tensorflow-${TF_VERSION} tensorflow

  # Install TF
  cd tensorflow
  echo -ne '\n' | ./configure
  bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package .
  pip install -U tensorflow-*.whl
  cd ..

  # Get tf.Text code
  curl -SsL -O https://github.com/tensorflow/text/archive/${TF_TEXT_VERSION}.zip
  unzip ${TF_TEXT_VERSION}.zip
  mv text-${TF_TEXT_VERSION} tensorflow_text
  cd tensorflow_text/
  ./oss_scripts/configure.sh

  # Create the tf.Text wheel
  bazel build oss_scripts/pip_package:build_pip_package
  ./bazel-bin/oss_scripts/pip_package/build_pip_package $HOME/wheels

  deactivate
done

cd
PYTHON_VERSION='cp36-cp36m'
PYTHON_BIN=/opt/python/${PYTHON_VERSION}/bin/python
$PYTHON_BIN -m pip install -U auditwheel==1.8.0 twine
$PYTHON_BIN -m pip install -U wheel==0.31.1
cd wheels
find . -type f -print | sed -e 's/^/auditwheel repair /' | sh

if [ "$UPLOAD_TO_PYPI" = upload ]; then
  cd wheelhouse
  twine upload *.whl
fi
