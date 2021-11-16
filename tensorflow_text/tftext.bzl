"""
Build rule for open source tf.text libraries.
"""

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
            deps = cc_op_kernels + select({
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
            deps = [
                ":" + library_name,
            ] + select({
                "@org_tensorflow//tensorflow:mobile": [
                    "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
                ],
                "//conditions:default": [],
            }),
        )

    native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        data = [":" + binary_name],
        deps = deps,
    )


def tf_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        tf_deps = [],
        copts = [],
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
        compatible_with: List of environments target can be built for
        testonly: If library is only for testing
        alwayslink: If symbols should be exported
    """
    if "kernel" in name:
        alwayslink = 1
    # These are "random" deps likely needed by each library (http://b/142433427)
    oss_deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
    ]
    deps += select({
        "@org_tensorflow//tensorflow:mobile": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:tf_header_lib",
        ] + tf_deps + oss_deps,
    })
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        compatible_with = compatible_with,
        testonly = testonly,
        alwayslink = alwayslink)


def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )


def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        "@org_tensorflow//tensorflow:macos": [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })


# buildozer: disable=function-docstring-args
def tf_pybind_extension(
        name,
        srcs,
        module_name,
        hdrs = [],
        features = [],
        srcs_version = "PY3",
        data = [],
        copts = [],
        linkopts = [],
        deps = [],
        defines = [],
        additional_exported_symbols = [],
        visibility = None,
        testonly = None,
        licenses = None,
        compatible_with = None,
        restricted_to = None,
        deprecation = None,
        link_in_framework = False,
        pytype_deps = [],
        pytype_srcs = []):
    """Builds a generic Python extension module."""
    _ignore = [module_name]
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ] + additional_exported_symbols

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.cc_binary(
        name = so_file,
        srcs = srcs + hdrs,
        data = data,
        copts = copts + [
            "-fno-strict-aliasing",
            "-fexceptions",
        ] + select({
            "@org_tensorflow//tensorflow:windows": [],
            "//conditions:default": [
                "-fvisibility=hidden",
            ],
        }),
        linkopts = linkopts + _rpath_linkopts(name) + select({
            "@org_tensorflow//tensorflow:macos": [
                # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                # not being exported.  There should be a better way to deal with this.
                "-Wl,-w",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "@org_tensorflow//tensorflow:windows": [],
            "//conditions:default": [
                "-Wl,--version-script",
                "$(location %s)" % version_script_file,
            ],
        }),
        deps = deps + [
            exported_symbols_file,
            version_script_file,
        ],
        defines = defines,
        features = features + ["-use_header_modules"],
        linkshared = 1,
        testonly = testonly,
        licenses = licenses,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.py_library(
        name = name,
        data = select({
            "//conditions:default": [so_file],
        }) + pytype_srcs,
        deps = pytype_deps,
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )



def pybind_extension(
        name = None,
        srcs = None,
        hdrs = None,
        data = None,
        visibility = None,
        deps = None,
        local_defines = None,
        module_name = None,
        additional_exported_symbols = []):
    """Creates a Python module implemented in C++.
    Python modules can depend on a py_extension. Other py_extensions can depend
    on a generated C++ library named with "_cc" suffix.
    Args:
      name: Name for this target.
      srcs: C++ source files.
      hdrs: C++ header files, for other py_extensions which depend on this.
      data: Files needed at runtime. This may include Python libraries.
      visibility: Controls which rules can depend on this.
      deps: Other C++ libraries that this library depends upon.
      local_defines: A list of custom definitions.
    """
    _ignore = [module_name, additional_exported_symbols]

    cc_library_name = name + "_cc"
    cc_binary_name = name + ".so"

    native.cc_library(
        name = cc_library_name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        visibility = visibility,
        deps = deps,
        alwayslink = True,
        local_defines = local_defines,
    )

    native.cc_binary(
        name = cc_binary_name,
        linkshared = True,
        linkstatic = True,
        # Ensure that the init function is exported. Required for gold.
        linkopts = ['-Wl,--export-dynamic-symbol=PyInit_{}'.format(name)],
        visibility = ["//visibility:private"],
        deps = [cc_library_name],
    )

    native.py_library(
        name = name,
        data = [cc_binary_name],
        visibility = visibility,
    )
