"""
Build rules for open source tf.text libraries.
"""
load("@rules_python//python:py_library.bzl", _py_library = "py_library")
load(
    "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    _pybind_extension = "pybind_extension",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_library = "pywrap_library",
)

def _local_tf_op_shim():
    """Returns a package-local `tf_op_shim` built against the TF wheel headers.

    Only the three shim sources are compiled here. They are linked dynamically
    against libtensorflow_framework.so (like any TF custom op), instead of
    depending on @org_tensorflow//tensorflow/lite/kernels/shim:tf_op_shim, which
    statically pulls TensorFlow implementation targets (protos, protobuf
    runtime, lib_internal, ...) into every _<op>.so.
    """
    name = "tf_op_shim_oss"
    if not native.existing_rule(name):
        tf_cc_library(
            name = name,
            srcs = [
                "@org_tensorflow//tensorflow/lite/kernels/shim:shape.cc",
                "@org_tensorflow//tensorflow/lite/kernels/shim:tf_op_shim.cc",
                "@org_tensorflow//tensorflow/lite/kernels/shim:tf_tensor_view.cc",
            ],
            deps = [
                "@com_google_absl//absl/algorithm:container",
                "@com_google_absl//absl/types:span",
            ],
        )
    return ":" + name

def _local_shape_inference_testutil():
    """Returns a package-local `shape_inference_testutil` built against the TF wheel headers."""
    name = "shape_inference_testutil_oss"
    if not native.existing_rule(name):
        hdr_out = "shape_inference_testutil_include/tensorflow/core/framework/shape_inference_testutil.h"
        native.genrule(
            name = name + "_hdr",
            srcs = ["@org_tensorflow//tensorflow/core/framework:shape_inference_testutil.h"],
            outs = [hdr_out],
            cmd = "cp $< $@",
            testonly = 1,
        )
        tf_cc_library(
            name = name,
            srcs = [
                "@org_tensorflow//tensorflow/core/framework:shape_inference_testutil.cc",
            ],
            hdrs = [hdr_out],
            includes = ["shape_inference_testutil_include"],
            testonly = 1,
        )
    return ":" + name

def py_tf_text_library(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        cc_op_defs = [],
        cc_op_kernels = []):
    """Creates build rules for TF.Text ops as shared libraries.

    Defines three targets:

    <name>
        Python library that exposes all ops defined in `cc_op_defs` and `py_srcs`.
    <name>_cc
        C++ library that registers any c++ ops in `cc_op_defs`, and includes the
        kernels from `cc_op_kernels`.
    python/ops/_<name>.so
        Shared library exposing the <name>_cc library.

    Args:
      name: The name for the python library target build by this rule.
      srcs: Python source files for the Python library.
      deps: Dependencies for the Python library.
      visibility: Visibility for the Python library.
      cc_op_defs: A list of c++ src files containing REGISTER_OP definitions.
      cc_op_kernels: A list of c++ targets containing kernels that are used
          by the Python library.
    """
    binary_path = "python/ops"
    if srcs:
        binary_path_end_pos = srcs[0].rfind("/")
        binary_path = srcs[0][0:binary_path_end_pos]
    binary_name = binary_path + "/_" + cc_op_kernels[0][1:] + ".so"
    if cc_op_defs:
        binary_name = binary_path + "/_" + name + ".so"
        library_name = name + "_cc"
        native.cc_library(
            name = library_name,
            srcs = cc_op_defs,
            copts = select({
                # Android supports pthread natively, -pthread is not needed.
                "@org_tensorflow//tensorflow:mobile": [],
                "//conditions:default": ["-pthread"],
            }),
            alwayslink = 1,
            deps = cc_op_kernels +
                   [_local_tf_op_shim()] +
                   select({
                       "@org_tensorflow//tensorflow:mobile": [
                           "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
                       ],
                       "//conditions:default": [],
                   }),
        )

        native.cc_binary(
            name = binary_name,
            copts = select({
                "@org_tensorflow//tensorflow:mobile": [],
                "//conditions:default": ["-pthread"],
            }),
            linkshared = 1,
            linkopts = select({
                "@org_tensorflow//tensorflow:macos": [
                    "-Wl,-exported_symbols_list,$(location //tensorflow_text:exported_symbols.lds)",
                ],
                "@org_tensorflow//tensorflow:windows": [],
                "//conditions:default": [
                    "-Wl,--version-script,$(location //tensorflow_text:version_script.lds)",
                ],
            }),
            deps = [
                ":" + library_name,
                "//tensorflow_text:version_script.lds",
                "//tensorflow_text:exported_symbols.lds",
            ] + select({
                "@org_tensorflow//tensorflow:mobile": [
                    "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
                ],
                "//conditions:default": [],
            }),
        )

    py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        data = [":" + binary_name],
        deps = deps,
    )


_OSS_ABSL_DEPS = [
    "@com_google_absl//absl/base",
    "@com_google_absl//absl/container:btree",
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/functional:any_invocable",
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:check",
    "@com_google_absl//absl/log:log",
    "@com_google_absl//absl/meta:type_traits",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/strings:cord",
    "@com_google_absl//absl/time",
    "@com_google_absl//absl/types:variant",
]

_SHARED_TFLITE_DEPS = [
    "@org_tensorflow//tensorflow/core/util:ragged_to_dense_util_common",
    "@org_tensorflow//tensorflow/lite:framework",
    "@org_tensorflow//tensorflow/lite:mutable_op_resolver",
    "@org_tensorflow//tensorflow/lite/c:common",
    "@org_tensorflow//tensorflow/lite/kernels/shim:tflite_op_shim",
    "@org_tensorflow//tensorflow/lite/kernels/shim:tflite_op_wrapper",
]

_TFLITE_OSS_ABSL_DEPS = [
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/strings:cord",
    "@com_google_absl//absl/time",
    "@com_google_absl//absl/types:variant",
]

def _dedupe_deps(deps, extra_deps):
    oss_deps = []
    for d in extra_deps:
        oss_deps.extend(_dedupe(deps, d))
    return oss_deps

def _dedupe(list, item):
    if item not in list:
        return [item]
    else:
        return []

def tf_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        tf_deps = [],
        copts = [],
        includes = [],
        compatible_with = None,
        testonly = 0,
        alwayslink = 0):
    """ A rule to build a TensorFlow library or OpKernel.

    Just like cc_library, but:
      * Adds alwayslink=1 for kernels (name has kernel in it)
      * Separates out TF deps for when building for Android.

    Args:
        name: Name of library
        srcs: Source files
        hdrs: Headers files
        deps: All non-TF dependencies
        tf_deps: All TF depenedencies
        copts: C options
        includes: Include dirs added to the compile line of the target and its dependents
        compatible_with: List of environments target can be built for
        testonly: If library is only for testing
        alwayslink: If symbols should be exported
    """
    if "kernel" in name:
        alwayslink = 1

    # These are "random" deps likely needed by each library (http://b/142433427)
    oss_deps = _dedupe_deps(deps, _OSS_ABSL_DEPS)

    deps = list(deps or []) + select({
        "@org_tensorflow//tensorflow:mobile": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ] + tf_deps + oss_deps,
    })
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        includes = includes,
        compatible_with = compatible_with,
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
    """ A rule to build a TensorFlow library or OpKernel.

    Args:
        name: Name of library
        srcs: Source files
        hdrs: Headers files
        deps: All non-TF dependencies
        copts: C options
        compatible_with: List of environments target can be built for
        testonly: If library is only for testing
        alwayslink: If symbols should be exported
    """

    tflite_deps = _SHARED_TFLITE_DEPS
    oss_deps = _TFLITE_OSS_ABSL_DEPS
    deps = list(deps or []) + tflite_deps + select({
        "@org_tensorflow//tensorflow:mobile": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ] + oss_deps,
    })
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        compatible_with = compatible_with,
        testonly = testonly,
        alwayslink = alwayslink,
    )

def tf_cc_test(
        name,
        srcs = [],
        deps = [],
        data = [],
        copts = [],
        size = None,
        timeout = None,
        tags = [],
        visibility = None,
        features = [],
        compatible_with = None,
        **kwargs):
    """A rule to build a TensorFlow C++ test.

    Args:
        name: Name of the test target.
        srcs: Source files
        deps: Test dependencies, including TensorFlow targets.
        data: Data files
        copts: C options
        size: Test size
        timeout: Test timeout
        tags: Tags
        visibility: Visibility
        features: Features
        compatible_with: List of environments target can be built for
        **kwargs: Additional keyword arguments forwarded to cc_test.
    """
    oss_deps = _dedupe_deps(deps, _OSS_ABSL_DEPS + [
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ])

    # Injected by copy.bara.sky when rewriting shape_inference_testutil deps.
    if ":shape_inference_testutil_oss" in (deps or []):
        _local_shape_inference_testutil()
    deps = list(deps or []) + [
        _local_tf_op_shim(),
    ] + select({
        "@org_tensorflow//tensorflow:mobile": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ] + oss_deps,
    })
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        copts = copts,
        size = size,
        timeout = timeout,
        tags = tags,
        visibility = visibility,
        features = features,
        compatible_with = compatible_with,
        **kwargs
    )

def tflite_cc_test(
        name,
        srcs = [],
        deps = [],
        data = [],
        copts = [],
        size = None,
        timeout = None,
        tags = [],
        visibility = None,
        features = [],
        compatible_with = None,
        **kwargs):
    """A rule to build a TensorFlow Lite C++ test.

    Args:
        name: Name of the test target.
        srcs: Source files
        deps: Test dependencies, including TensorFlow targets.
        data: Data files
        copts: C options
        size: Test size
        timeout: Test timeout
        tags: Tags
        visibility: Visibility
        features: Features
        compatible_with: List of environments target can be built for
        **kwargs: Additional keyword arguments forwarded to cc_test.
    """
    tflite_deps = _SHARED_TFLITE_DEPS + [
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ]
    oss_deps = _dedupe_deps(deps, _TFLITE_OSS_ABSL_DEPS + ["@com_google_googletest//:gtest", "@com_google_googletest//:gtest_main"])

    deps = list(deps or []) + tflite_deps + select({
        "@org_tensorflow//tensorflow:mobile": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ] + oss_deps,
    })
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        copts = copts,
        size = size,
        timeout = timeout,
        tags = tags,
        visibility = visibility,
        features = features,
        compatible_with = compatible_with,
        **kwargs
    )

def py_library(name, lazy_imports = False, strict_deps = None, **kwargs):
    _ = (lazy_imports, strict_deps)  # buildifier: disable=unused-variable
    _py_library(
        name = name,
        **kwargs
    )

def pybind_extension(name, deps = None, **kwargs):
    deps = deps or []
    deps = deps + ["@pybind11//:pybind11"]
    _pybind_extension(
        name = name,
        deps = deps,
        **kwargs
    )

def if_pywrap(if_true = None, if_false = None):
    _ = (if_false,)  # buildifier: disable=unused-variable

    # Always use pywrap.
    return if_true or []

def pywrap_library(name, **kwargs):
    _pywrap_library(
        name = name,
        **kwargs
    )

def pywrap_binaries(name, **kwargs):
    _pywrap_binaries(
        name = name,
        **kwargs
    )
