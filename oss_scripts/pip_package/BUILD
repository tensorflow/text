load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@org_tensorflow//third_party/xla/third_party/py:python_wheel.bzl", "collect_data_files", "transitive_py_deps")

# Tools for building the TF.Text pip package.
load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("//oss_scripts/pip_package:wheel.bzl", "tensorflow_text_wheel")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    generate_hashes = True,
    requirements_in = "@release_or_nightly//:requirements.in",
    requirements_txt = REQUIREMENTS,
)

py_binary(
    name = "tensorflow_build_info",
    srcs = ["tensorflow_build_info.py"],
    deps = [
        "@absl_py//absl:app",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

string_flag(
    name = "output_path",
    build_setting_default = "dist",
)

py_binary(
    name = "build_wheel_py",
    srcs = ["build_wheel.py"],
    main = "build_wheel.py",
    deps = [
        #":build_utils",
        #"@bazel_tools//tools/python/runfiles",
        #"@pypi//build",
        #"@pypi//setuptools",
        #"@pypi//wheel",
    ],
)

filegroup(
    name = "wheel_sources",
    srcs = [
        "LICENSE",
        "MANIFEST.in",
        "setup.nightly.py",
        ":transitive_data_deps",
        ":transitive_py_deps",
    ],
)

transitive_py_deps(
    name = "transitive_py_deps",
    deps = ["//tensorflow_text"],
)

collect_data_files(
    name = "transitive_data_deps",
    deps = ["//tensorflow_text"],
)

tensorflow_text_wheel(
    name = "tensorflow_text_wheel",
    srcs = [":wheel_sources"],
)

#sh_binary(
#    name = "build_pip_package",
#    srcs = ["build_pip_package.sh"],
#    data = [
#        "LICENSE",
#        "MANIFEST.in",
#        "setup.nightly.py",
#        "setup.py",
#        "//tensorflow_text",
#    ],
#)
