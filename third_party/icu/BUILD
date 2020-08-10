# This empty BUILD file is required to make Bazel treat this directory as a package.

load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

bzl_library(
    name = "data/BUILD_bzl",
    srcs = ["data/BUILD.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "BUILD_bzl",
    srcs = ["BUILD.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
