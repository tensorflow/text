// Copyright 2020 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace text {

REGISTER_OP("CaseFoldUTF8")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Applies case folding to every UTF8 string in input_tensor. The input is a dense
tensor of any shape and the output has the same shape as the input.

For example if:

  input = [ 'The   Quick-Brown',
            'CAT jumped over',
            'the lazy dog  !!  ']

  output = [ 'The   quick-brown',
             'cat jumped over',
             'the lazy dog  !!  ']
)doc");

REGISTER_OP("NormalizeUTF8")
    .Input("input: string")
    .Attr("normalization_form: string")
    .Output("output: string")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Normalizes each UTF8 string in the input tensor using 'normalization_form'
rules.

See http://unicode.org/reports/tr15/
)doc");

}  // namespace text
}  // namespace tensorflow
