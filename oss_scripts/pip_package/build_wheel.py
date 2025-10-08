# coding=utf-8
# Copyright 2025 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script that builds a tf text wheel, intended to be run via bazel."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)
parser.add_argument(
    "--platform",
    default="",
    required=False,
    help="Platform name to be passed to setup.py",
)
args = parser.parse_args()


def copy_file(
    src_file: str,
    dst_dir: str,
) -> None:
  """Copy a file to the destination directory.

  Args:
    src_file: file to be copied
    dst_dir: destination directory
  """

  dest_dir_path = os.path.join(dst_dir, os.path.dirname(src_file))
  os.makedirs(dest_dir_path, exist_ok=True)
  shutil.copy(src_file, dest_dir_path)
  os.chmod(os.path.join(dst_dir, src_file), 0o644)


def prepare_srcs(deps: list[str], srcs_dir: str) -> None:
  """Filter the sources and copy them to the destination directory.

  Args:
    deps: a list of paths to files.
    srcs_dir: target directory where files are copied to.
  """

  for file in deps:
    print(file)
    if not (file.startswith("bazel-out") or file.startswith("external")):
      copy_file(file, srcs_dir)


def build_wheel(
    dir_path: str,
    cwd: str,
    platform: str,
) -> None:
  """Build the wheel in the target directory.

  Args:
    dir_path: directory where the wheel will be stored
    cwd: path to directory with wheel source files
    platform: platform name to pass to setup.py.
  """

  subprocess.run(
      [
          sys.executable,
          "setup.nightly.py",
          "bdist_wheel",
          f"--dist-dir={dir_path}",
          f"--plat-name={platform}",
      ],
      check=True,
      cwd=cwd,
  )


tmpdir = tempfile.TemporaryDirectory(prefix="tensorflow_text")
sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_srcs(args.srcs, pathlib.Path(sources_path))
  build_wheel(
      os.path.join(os.getcwd(), args.output_path),
      tmpdir.path,
      args.platform,
  )
finally:
  if tmpdir:
    tmpdir.cleanup()
