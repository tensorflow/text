"""Wrapper around rules_python py_library that accepts (and ignores) strict_deps."""

load("@rules_python//python:py_library.bzl", _py_library = "py_library")

def py_library(strict_deps = None, **kwargs):
    _py_library(**kwargs)
