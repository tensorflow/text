"""
Build rule for tf.text libraries.

BEGIN GOOGLE-INTERNAL
Example usage:

py_tf_text_library(
    name = "string_ops",
    srcs = ["python/ops/string_ops.py"],
    cc_op_defs = ["core/ops/string_ops.cc"],
    cc_op_kernels = ["//third_party/tensorflow_text/core/kernels:string_kernels"],
    deps = ["//third_party/tensorflow/python/framework:dtypes"],
)
END GOOGLE-INTERNAL
"""

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py", "tf_opts_nortti_if_mobile")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")

# Generates a Python library for the ops defined in `op_defs` and `py_srcs`.
#
# Defines three targets:
#
# <name>
#     Python library that exposes all ops defined in `cc_op_defs` and `py_srcs`.
# <name>_cc
#     C++ library that registers any c++ ops in `cc_op_defs`, and includes the
#     kernels from `cc_op_kernels`.
# gen_<name>_py
#     Python library that exposes any c++ ops.
#
# Args:
#   name: The name for the python library target build by this rule.
#   srcs: Python source files for the Python library.
#   deps: Dependencies for the Python library.
#   compatible_with: Standard blaze cc_library for the cc_library.
#   visibility: Visibility for the Python library.
#   cc_op_defs: A list of c++ src files containing REGISTER_OP definitions.
#   cc_op_kernels: A list of c++ targets containing kernels that are used
#       by the Python library.
def py_tf_text_library(
        name,
        srcs = [],
        deps = [],
        compatible_with = None,
        visibility = None,
        cc_op_defs = [],
        cc_op_kernels = []):
    if cc_op_defs:
        # C++ library that registers ops.
        op_def_lib_name = name + "_cc"
        cc_library(
            name = op_def_lib_name,
            srcs = cc_op_defs,
            deps = cc_op_kernels +
                   ["@org_tensorflow//tensorflow/lite/kernels/shim:tf_op_shim"] +
                   select({
                       "@org_tensorflow//tensorflow:mobile": [
                           "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
                       ],
                       "//conditions:default": [
                           "@org_tensorflow//tensorflow/core:framework",
                           "@org_tensorflow//tensorflow/core:framework_headers_lib",
                       ],
                   }),
            features = select({
                "@org_tensorflow//tensorflow:android": [
                    "-layering_check",
                ],
                "//conditions:default": [],
            }),
            alwayslink = 1,
        )
        deps = deps + [":" + op_def_lib_name]

        # Python wrapper that exposes c++ ops.
        gen_py_lib_name = "gen_" + name + "_py"
        gen_py_out = "gen_" + name + ".py"
        tf_gen_op_wrapper_py(
            name = gen_py_lib_name,
            out = gen_py_out,
            deps = [":" + op_def_lib_name],
        )
        deps = deps + [":" + gen_py_lib_name]

    # Python library that exposes all ops.
    py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        deps = cc_op_kernels + deps + [
            "@org_tensorflow//tensorflow/python/framework:for_generated_wrappers",
        ],
    )

# Enable build_cleaner to update py_tf_text_library targets.
# GOOGLE-INTERNAL: See: go/build-cleaner-build-extensions
# register_extension_info(
#     extension = py_tf_text_library,
#     label_regex_for_dep = "{extension_name}",
# )

def tf_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        tf_deps = [],
        copts = [],
        features = [],
        compatible_with = None,
        testonly = 0,
        alwayslink = 0):
    """ A rule to build a TensorFlow library or OpKernel.

    Just like cc_library, but:
      * Adds alwayslink=1 for kernels (name has kernel in it)
      * Passes -DGOOGLE_CUDA=1 if we're building with --config=cuda.
      * Separates out TF deps for when building for Android.

    Args:
        name: Name of library
        srcs: Source files
        hdrs: Headers files
        deps: All non-TF dependencies
        tf_deps: All TF depenedencies
        copts: C options
        compatible_with: List of environments target can be built for
        testonly: If library is only for testing
        alwayslink: If symbols should be exported.
    """
    if "kernel" in name:
        alwayslink = 1
    if tf_deps:
        deps += select({
            "@org_tensorflow//tensorflow:mobile": [
                "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
            ],
            "//conditions:default": tf_deps,
        })
        features += select({
            "@org_tensorflow//tensorflow:android": [
                "-layering_check",
            ],
            "//conditions:default": [],
        })
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts + tf_opts_nortti_if_mobile(),
        features = features,
        testonly = testonly,
        alwayslink = alwayslink,
    )

def tflite_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        compatible_with = None,
        testonly = 0,
        alwayslink = 0):
    """ A rule to build a TensorFlow Lite library or OpKernel.

    Args:
        name: Name of library
        srcs: Source files
        hdrs: Headers files
        deps: All non-TF dependencies
        copts: C options
        compatible_with: List of environments target can be built for
        testonly: If library is only for testing
        alwayslink: If symbols should be exported.
    """
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts + tf_opts_nortti_if_mobile(),
        testonly = testonly,
        alwayslink = alwayslink,
    )

# Allow build_cleaner to manage dependencies of tf_cc_library build rules.
# GOOGLE-INTERNAL: See: go/build-cleaner-build-extensions
# register_extension_info(
#     extension = tf_cc_library,
#     label_regex_for_dep = "{extension_name}",
# )

def extra_py_deps():
    return []
