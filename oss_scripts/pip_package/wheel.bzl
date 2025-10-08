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

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@tensorflow_text_wheel//:wheel.bzl", "WHEEL_VERSION")
load("@tf_wheel_version_suffix//:wheel_version_suffix.bzl", "WHEEL_VERSION_SUFFIX")

def _get_full_wheel_name(
        wheel_version,
        is_nightly):
    wheel_name_template = "tensorflow_text{nightly_suffix}-{wheel_version}-py{major_python_version}-none-any.whl"
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    return wheel_name_template.format(
        major_python_version = python_version[0],
        wheel_version = wheel_version,
        nightly_suffix = "_nightly" if is_nightly else "",
    )

def _is_nightly_build():
    return WHEEL_VERSION_SUFFIX.startswith(".dev") and not "selfbuilt" in WHEEL_VERSION_SUFFIX

def _is_selfbuilt_build():
    return WHEEL_VERSION_SUFFIX.startswith(".dev") and "selfbuilt" in WHEEL_VERSION_SUFFIX

def _wheel_impl(ctx):
    executable = ctx.executable.wheel_binary
    output_path = ctx.attr.output_path[BuildSettingInfo].value
    is_nightly = _is_nightly_build()

    full_wheel_version = (WHEEL_VERSION + WHEEL_VERSION_SUFFIX)

    args = ctx.actions.args()
    name = _get_full_wheel_name(
        wheel_version = full_wheel_version,
        is_nightly = is_nightly,
    )
    output_file = ctx.actions.declare_file(output_path + "/" + name)
    outputs = [output_file]
    srcs = []
    for src in ctx.attr.source_files:
        for f in src.files.to_list():
            srcs.append(f)
            args.add("--srcs=%s" % (f.path))
    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = srcs,
        outputs = outputs,
        executable = executable,
        mnemonic = "BuildTensorflowTextWheel",
    )

    return [DefaultInfo(files = depset(direct = [output_file]))]

_wheel = rule(
    attrs = {
        "wheel_binary": attr.label(
            default = Label("//oss_scripts/pip_package:build_wheel_py"),
            executable = True,
            cfg = "exec",
        ),
        "source_files": attr.label_list(allow_files = True),
        "output_path": attr.label(default = Label("//oss_scripts/pip_package:output_path")),
    },
    implementation = _wheel_impl,
    executable = False,
)

def tensorflow_text_wheel(
        name,
        srcs = None):
    _wheel(
        name = name,
        source_files = srcs,
    )
