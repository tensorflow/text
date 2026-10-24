# Tools for building the TF.Text pip package.
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:py_binary.bzl", "py_binary")

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
        "@pypi//absl_py:pkg",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.py"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "setup.nightly.py",
        "setup.py",
        "//tensorflow_text",
        "//tensorflow_text/core/pybinds:pybinds_binaries",
    ],
    deps = [
        "@pypi//setuptools:pkg",
    ],
)
