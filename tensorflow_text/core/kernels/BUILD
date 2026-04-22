"""Kernels for tf.text ops.
All implementation files moved to //third_party/tensorflow/core/kernels/text.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_text:license"],
    default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = ["//visibility:public"],
)

exports_files(["LICENSE"])

# Aliases to relocated targets

cc_library(
    name = "wordpiece_tokenizer",
    hdrs = ["wordpiece_tokenizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:wordpiece_tokenizer"],
)

cc_library(
    name = "boise_offset_converter",
    hdrs = ["boise_offset_converter.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:boise_offset_converter"],
)

cc_library(
    name = "boise_offset_converter_kernel",
    hdrs = ["boise_offset_converter_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:boise_offset_converter_kernel"],
)

cc_library(
    name = "boise_offset_converter_kernel_template",
    hdrs = ["boise_offset_converter_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:boise_offset_converter_kernel_template"],
)

cc_library(
    name = "byte_splitter",
    hdrs = ["byte_splitter.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:byte_splitter"],
)

cc_library(
    name = "byte_splitter_kernel",
    hdrs = ["byte_splitter_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:byte_splitter_kernel"],
)

cc_library(
    name = "byte_splitter_kernel_template",
    hdrs = ["byte_splitter_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:byte_splitter_kernel_template"],
)

cc_library(
    name = "byte_splitter_tflite",
    hdrs = ["byte_splitter_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:byte_splitter_tflite"],
)

cc_library(
    name = "constrained_sequence",
    hdrs = ["constrained_sequence.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:constrained_sequence"],
)

cc_library(
    name = "fast_bert_normalizer",
    hdrs = ["fast_bert_normalizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer"],
)

cc_library(
    name = "fast_bert_normalizer_model",
    hdrs = ["fast_bert_normalizer_model_generated.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_model"],
)

cc_library(
    name = "fast_bert_normalizer_model_builder",
    hdrs = ["fast_bert_normalizer_model_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_model_builder"],
)

cc_library(
    name = "fast_bert_normalizer_kernel_template",
    hdrs = ["fast_bert_normalizer_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_kernel_template"],
)

cc_library(
    name = "fast_bert_normalizer_tf_kernel",
    hdrs = ["fast_bert_normalizer_tf_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_tf_kernel"],
)

cc_library(
    name = "fast_bert_normalizer_tflite",
    hdrs = ["fast_bert_normalizer_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_tflite"],
)

cc_library(
    name = "darts_clone_trie_builder",
    hdrs = ["darts_clone_trie_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:darts_clone_trie_builder"],
)

cc_library(
    name = "darts_clone_trie_wrapper",
    hdrs = ["darts_clone_trie_wrapper.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:darts_clone_trie_wrapper"],
)

cc_library(
    name = "disjoint_set_forest",
    hdrs = ["disjoint_set_forest.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:disjoint_set_forest"],
)

cc_library(
    name = "fast_wordpiece_tokenizer",
    hdrs = ["fast_wordpiece_tokenizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_model",
    hdrs = ["fast_wordpiece_tokenizer_model_generated.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_model"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_model_builder",
    hdrs = ["fast_wordpiece_tokenizer_model_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_model_builder"],
)

cc_library(
    name = "phrase_tokenizer_model",
    hdrs = ["phrase_tokenizer_model_generated.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_model"],
)

cc_library(
    name = "phrase_tokenizer_model_builder",
    hdrs = ["phrase_tokenizer_model_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_model_builder"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_kernel",
    hdrs = ["fast_wordpiece_tokenizer_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_kernel"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_kernel_template",
    hdrs = ["fast_wordpiece_tokenizer_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_kernel_template"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_tflite",
    hdrs = ["fast_wordpiece_tokenizer_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_tflite"],
)

cc_library(
    name = "fast_wordpiece_tokenizer_utils",
    hdrs = ["fast_wordpiece_tokenizer_utils.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_utils"],
)

cc_library(
    name = "mst_solver",
    hdrs = ["mst_solver.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:mst_solver"],
)

cc_library(
    name = "ngrams_kernel_template",
    hdrs = ["ngrams_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:ngrams_kernel_template"],
)

cc_library(
    name = "ngrams_kernel",
    hdrs = ["ngrams_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:ngrams_kernel"],
)

cc_library(
    name = "ngrams_tflite",
    hdrs = ["ngrams_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:ngrams_tflite"],
)

cc_library(
    name = "ragged_tensor_to_tensor_tflite",
    hdrs = ["ragged_tensor_to_tensor_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:ragged_tensor_to_tensor_tflite"],
)

cc_library(
    name = "regex_split",
    hdrs = ["regex_split.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:regex_split"],
)

cc_library(
    name = "round_robin_trimmer",
    hdrs = ["round_robin_trimmer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer"],
)

cc_library(
    name = "round_robin_trimmer_kernel",
    hdrs = ["round_robin_trimmer_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer_kernel"],
)

cc_library(
    name = "round_robin_trimmer_kernel_template",
    hdrs = ["round_robin_trimmer_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer_kernel_template"],
)

cc_library(
    name = "round_robin_trimmer_tflite",
    hdrs = ["round_robin_trimmer_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer_tflite"],
)

cc_library(
    name = "sentence_breaking_utils",
    hdrs = ["sentence_breaking_utils.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_breaking_utils"],
)

cc_library(
    name = "sentence_fragmenter",
    hdrs = ["sentence_fragmenter.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter"],
)

cc_library(
    name = "sentence_fragmenter_v2",
    hdrs = ["sentence_fragmenter_v2.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2"],
)

cc_library(
    name = "sentence_fragmenter_v2_kernel",
    hdrs = ["sentence_fragmenter_v2_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2_kernel"],
)

cc_library(
    name = "sentence_fragmenter_v2_kernel_template",
    hdrs = ["sentence_fragmenter_v2_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2_kernel_template"],
)

cc_library(
    name = "sentence_fragmenter_v2_tflite",
    hdrs = ["sentence_fragmenter_v2_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2_tflite"],
)

cc_library(
    name = "spanning_tree_iterator",
    testonly = 1,
    hdrs = ["spanning_tree_iterator.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:spanning_tree_iterator"],
)

cc_library(
    name = "text_kernels_test_util",
    testonly = 1,
    hdrs = ["text_kernels_test_util.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:text_kernels_test_util"],
)

cc_library(
    name = "tflite_ops",
    hdrs = [
        "byte_splitter_tflite.h",
        "fast_bert_normalizer_tflite.h",
        "fast_wordpiece_tokenizer_tflite.h",
        "ngrams_tflite.h",
        "ragged_tensor_to_tensor_tflite.h",
        "round_robin_trimmer_tflite.h",
        "sentence_fragmenter_v2_tflite.h",
        "utf8_binarize_tflite.h",
        "whitespace_tokenizer_tflite.h",
    ],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:tflite_ops"],
)

cc_library(
    name = "trimmer",
    hdrs = ["trimmer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:trimmer"],
)

cc_library(
    name = "whitespace_tokenizer",
    hdrs = ["whitespace_tokenizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer"],
)

cc_library(
    name = "whitespace_tokenizer_kernel",
    hdrs = ["whitespace_tokenizer_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_kernel"],
)

cc_library(
    name = "whitespace_tokenizer_kernel_template",
    hdrs = ["whitespace_tokenizer_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_kernel_template"],
)

cc_library(
    name = "whitespace_tokenizer_tflite",
    hdrs = ["whitespace_tokenizer_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_tflite"],
)

cc_library(
    name = "whitespace_tokenizer_config_builder",
    hdrs = ["whitespace_tokenizer_config_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_config_builder"],
)

cc_library(
    name = "string_vocab",
    hdrs = ["string_vocab.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:string_vocab"],
)

cc_library(
    name = "phrase_tokenizer",
    hdrs = ["phrase_tokenizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer"],
)

cc_library(
    name = "phrase_tokenizer_kernel_template",
    hdrs = ["phrase_tokenizer_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_kernel_template"],
)

cc_library(
    name = "phrase_tokenizer_kernel",
    hdrs = ["phrase_tokenizer_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_kernel"],
)

cc_library(
    name = "utf8_binarize",
    hdrs = ["utf8_binarize.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize"],
)

cc_library(
    name = "utf8_binarize_kernel_template",
    hdrs = ["utf8_binarize_kernel_template.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize_kernel_template"],
)

cc_library(
    name = "utf8_binarize_kernel",
    hdrs = ["utf8_binarize_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize_kernel"],
)

cc_library(
    name = "utf8_binarize_tflite",
    hdrs = ["utf8_binarize_tflite.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize_tflite"],
)

alias(
    name = "boise_offset_converter_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:boise_offset_converter_test",
)

alias(
    name = "byte_splitter_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:byte_splitter_test",
)

alias(
    name = "constrained_sequence_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:constrained_sequence_kernel",
)

alias(
    name = "constrained_sequence_kernel_input_validation_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:constrained_sequence_kernel_input_validation_test",
)

alias(
    name = "exp_greedy_constrained_sequence_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:exp_greedy_constrained_sequence_kernel_test",
)

alias(
    name = "exp_viterbi_constrained_sequence_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:exp_viterbi_constrained_sequence_kernel_test",
)

alias(
    name = "fast_bert_normalizer_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_test",
)

alias(
    name = "log_greedy_constrained_sequence_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:log_greedy_constrained_sequence_kernel_test",
)

alias(
    name = "log_viterbi_constrained_sequence_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:log_viterbi_constrained_sequence_kernel_test",
)

alias(
    name = "darts_clone_trie_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:darts_clone_trie_test",
)

alias(
    name = "disjoint_set_forest_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:disjoint_set_forest_test",
)

alias(
    name = "fast_wordpiece_tokenizer_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_test",
)

alias(
    name = "fast_wordpiece_tokenizer_utils_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_utils_test",
)

alias(
    name = "mst_op_kernels",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:mst_op_kernels",
)

alias(
    name = "mst_solver_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:mst_solver_test",
)

alias(
    name = "mst_solver_random_comparison_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:mst_solver_random_comparison_test",
)

alias(
    name = "edit_changes_proto",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:edit_changes_proto",
)

alias(
    name = "edit_changes_cc_proto",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:edit_changes_cc_proto",
)

alias(
    name = "ngrams_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:ngrams_kernel_test",
)

alias(
    name = "ngrams_tflite_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:ngrams_tflite_test",
)

alias(
    name = "normalize_kernels",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:normalize_kernels",
)

alias(
    name = "ragged_tensor_to_tensor_tflite_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:ragged_tensor_to_tensor_tflite_test",
)

alias(
    name = "regex_split_kernels",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:regex_split_kernels",
)

alias(
    name = "regex_split_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:regex_split_test",
)

alias(
    name = "round_robin_trimmer_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer_test",
)

alias(
    name = "rouge_l_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:rouge_l_kernel",
)

alias(
    name = "rouge_l_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:rouge_l_kernel_test",
)

alias(
    name = "sentence_breaking_kernels",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:sentence_breaking_kernels",
)

alias(
    name = "sentence_breaking_utils_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:sentence_breaking_utils_test",
)

alias(
    name = "sentence_fragmenter_v2_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2_test",
)

alias(
    name = "sentencepiece_kernels",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:sentencepiece_kernels",
)

alias(
    name = "spanning_tree_iterator_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:spanning_tree_iterator_test",
)

alias(
    name = "split_merge_tokenize_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:split_merge_tokenize_kernel",
)

alias(
    name = "tokenizer_from_logits_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:tokenizer_from_logits_kernel",
)

alias(
    name = "unicode_script_tokenize_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:unicode_script_tokenize_kernel",
)

alias(
    name = "unicode_script_tokenize_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:unicode_script_tokenize_kernel_test",
)

alias(
    name = "whitespace_tokenize_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenize_kernel",
)

alias(
    name = "whitespace_tokenize_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenize_kernel_test",
)

alias(
    name = "whitespace_tokenizer_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_test",
)

alias(
    name = "whitespace_tokenizer_config_builder_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_config_builder_test",
)

alias(
    name = "phrase_tokenizer_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_test",
)

alias(
    name = "utf8_binarize_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize_test",
)

alias(
    name = "wordpiece_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:wordpiece_kernel",
)

alias(
    name = "wordpiece_kernel_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text:wordpiece_kernel_test",
)
