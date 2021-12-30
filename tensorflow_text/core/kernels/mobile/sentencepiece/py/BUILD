load("//devtools/clif/python:clif_build_rule.bzl", "py_clif_cc")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["notice"])

# Visibility rules
package(default_visibility = [
    "//tensorflow_text:__subpackages__",
])

py_clif_cc(
    name = "model_converter",
    srcs = [
        "model_converter.clif",
    ],
    deps = [
        "//tensorflow_text/core/kernels/mobile/sentencepiece:model_converter",
    ],
)

pybind_extension(
    name = "pywrap_tflite_registerer",
    srcs = [
        "pywrap_tflite_registerer.cc",
    ],
    additional_exported_symbols = ["TFLite_SentencepieceTokenizerRegisterer"],
    module_name = "pywrap_tflite_registerer",
    srcs_version = "PY3ONLY",
    deps = [
        "@pybind11",
        "//third_party/python_runtime:headers",
        # lite:framework tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
        "//tensorflow_text/core/kernels/mobile/sentencepiece:py_tflite_registerer",
    ],
)

pybind_extension(
    name = "pywrap_model_converter",
    srcs = ["pywrap_model_converter.cc"],
    additional_exported_symbols = [
        "ConvertSentencepieceModel",
        "ConvertSentencepieceModelForDecoder",
        "GetVocabularySize",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_model_converter",
    srcs_version = "PY3ONLY",
    deps = [
        "@com_google_absl//absl/status",
        "@pybind11",
        "//third_party/python_runtime:headers",
        # lite:framework tensorflow dep,
        "//third_party/tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_text/core/kernels/mobile/sentencepiece:model_converter",
    ],
)
