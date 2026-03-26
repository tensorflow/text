"""Wrapper around rules_python py_test that accepts (and ignores) strict_deps."""

load("@rules_python//python:py_test.bzl", _py_test = "py_test")

def py_test(strict_deps = None, **kwargs):
    _py_test(**kwargs)
