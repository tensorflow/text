# coding=utf-8
# Copyright 2020 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarking utils for TF.Text ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow_datasets as tfds

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import benchmark


class OpBenchmark(benchmark.Benchmark):
  """Base class for op benchmarks."""

  def __init__(self):
    super(OpBenchmark, self).__init__()
    self.input_data = None
    self.iterator = None

  def load_input_data(self, batch_size):
    """Loads the IMDB dataset and sets up the input data to run the ops on."""

    data = tfds.load(
        'imdb_reviews/plain_text', split=tfds.Split.TRAIN).batch(batch_size)
    # The input data has shape [batch_size, data] and the op is run multiple
    # iterations over the first batch
    if context.executing_eagerly():
      self.iterator = data.as_numpy_iterator()
      self.input_data = [x['text'] for x in self.iterator][0]
    else:
      self.iterator = dataset_ops.make_initializable_iterator(data)
      self.input_data = self.iterator.get_next()['text']

  def run_and_report(self,
                     fn,
                     iters,
                     burn_iters,
                     benchmark_name,
                     use_tf_function=False,
                     **kwargs):
    """Runs the benchmark and reports results.

    Arguments:
      fn: Function to be benchmarked.
      iters: Number of iterations to run the benchmark.
      burn_iters: Number of warm-up iterations to run to reach a stable state.
      benchmark_name: Name used for reporting the results.
      use_tf_function: Bool, specifies whether the function should be wrapped in
        a @tf.function while running eagerly. By default 'False' and ignored in
        graph mode.
      **kwargs: Kwargs to the benchmarked function.

    Returns:
      Dict which contains the wall time report for the runned op.
    """
    if context.executing_eagerly():
      self._run_and_report_eagerly(fn, iters, burn_iters, benchmark_name,
                                   use_tf_function, **kwargs)
    else:
      self._run_and_report_graphmode(fn, iters, burn_iters, benchmark_name,
                                     **kwargs)

  def _run_and_report_eagerly(self,
                              fn,
                              iters,
                              burn_iters,
                              benchmark_name,
                              use_tf_function=False,
                              **kwargs):
    """Runs and reports benchmarks eagerly."""
    if self.input_data is None:
      raise ValueError(
          'Input data is missing for {} benchmark'.format(benchmark_name))

    @def_function.function
    def tf_func():
      fn(self.input_data, **kwargs)

    def func():
      fn(self.input_data, **kwargs)

    op = tf_func if use_tf_function else func

    for _ in range(burn_iters):
      op()

    total_time = 0
    for _ in range(iters):
      start = time.time()
      op()
      total_time += time.time() - start

    mean_time = total_time / iters
    benchmark_name += '_function' if use_tf_function else ''
    self.report_benchmark(
        iters=iters, wall_time=mean_time, name=benchmark_name + '_eager')

  def _run_and_report_graphmode(self, fn, iters, burn_iters, benchmark_name,
                                **kwargs):
    """Runs and reports benchmarks in graph mode."""
    if self.input_data is None:
      raise ValueError(
          'Input data is missing for {} benchmark'.format(benchmark_name))

    if self.iterator is None:
      raise ValueError(
          'Input iterator is missing and could not be initialized for {}'
          ' benchmark'.format(benchmark_name))

    # Uses the benchmark config to disable the static graph optimizations
    with session.Session(config=benchmark.benchmark_config()) as sess:
      sess.run(self.iterator.initializer)
      sess.run(lookup_ops.tables_initializer())
      sess.run(variables_lib.global_variables_initializer())

      inputs = sess.run(self.input_data)
      benchmark_op = fn(inputs, **kwargs)
      self.run_op_benchmark(
          sess,
          benchmark_op,
          min_iters=iters,
          burn_iters=burn_iters,
          name=benchmark_name + '_graph')
