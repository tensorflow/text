"""
Build rule for tf.text libraries.
"""

def py_tf_text_library(
    name,
    srcs = [],
    deps = [],
    visibility = None,
    cc_op_defs = [],
    cc_op_kernels = []):
  """
  Creates build rules for TF.Text ops as shared libraries.
  """
  binary_name = "python/ops/_" + name + ".so"
  native.cc_binary(
      name = binary_name,
      srcs = cc_op_defs,
      copts = [ "-pthread", "-std=c++11", ],
      linkshared = 1,
      deps = cc_op_kernels + [
          "@local_config_tf//:libtensorflow_framework",
          "@local_config_tf//:tf_header_lib",
      ],
  )

  native.py_library(
      name = name,
      srcs = srcs,
      srcs_version = "PY2AND3",
      visibility = visibility,
      data = [ ":" + binary_name ],
      deps = deps,
  )
