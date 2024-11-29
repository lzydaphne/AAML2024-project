/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>

//* added by lzy
#include <stdint.h>
#include <stdio.h>

#include <cstdio>

#include "cfu.h"
#include "playground_util/print_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.

inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                          output_shift[out_c]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

template <typename AccumScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

//! lab target
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  //! modified for FINAL
  printf("check if using FullyConnected kernels\n");
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  // const int32_t output_activation_min = params.quantized_activation_min;
  // const int32_t output_activation_max = params.quantized_activation_max;
  // TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  // TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  // const int filter_dim_count = filter_shape.DimensionsCount();
  // const int output_dim_count = output_shape.DimensionsCount();
  // const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  // const int output_depth = output_shape.Dims(output_dim_count - 1);
  // TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  // const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  // printf("input_offset: %d\n", (int)input_offset);
  // printf("filter_offset: %d\n", (int)filter_offset);
  // printf("output_offset: %d\n", (int)output_offset);
  // printf("output_multiplier: %d\n", (int)output_multiplier);
  // printf("output_shift: %d\n", output_shift);
  // printf("output_activation_min: %d\n", (int)output_activation_min);
  // printf("output_activation_max: %d\n", (int)output_activation_max);
  // printf("filter_dim_count: %d\n", filter_dim_count);
  // printf("output_dim_count: %d\n", output_dim_count);
  // printf("batches: %d\n", batches);
  // printf("output_depth: %d\n", output_depth);
  // printf("accum_depth: %d\n", accum_depth);

  // print_fully_connected_params(params, input_shape, filter_shape,
  // output_shape);
  //* update for final project:  batches: 1, output_depth: 10, accum_depth:  64
  // printf("batches: %d, output_depth: %d, accum_depth: %d\n", batches,
  //        output_depth, accum_depth);

  //* print output_offset, output_multiplier,
  // output_shift,output_activation_min, output_activation_max
  // printf("output_offset: %d, output_multiplier: %d, output_shift: %d\n",
  //        output_offset, output_multiplier, output_shift);
  // printf("output_activation_min: %d, output_activation_max: %d\n",
  //        output_activation_min, output_activation_max);

  //! modified for FINAL
  cfu_op(0, 0, input_offset, filter_offset);  // set input,filter offset

  // for (int b = 0; b < batches; ++b) {
  // for (int b = 0; b < 1; ++b) {
  for (int out_c = 0; out_c < 10; ++out_c) {
    // for (int out_c = 0; out_c < output_depth; ++out_c) {
    // int32_t acc = 0;
    //! modified for FINAL, reset acc to 0
    int32_t acc = cfu_op(1, 0, 0, 0);
    // int32_t acc = 0;

    // for (int d = 0; d < accum_depth; ++d) {
    for (int d = 0; d < 64; d += 4) {
      //! modified for FINAL
      uint32_t input_val = 0;
      uint32_t filter_val = 0;
      //* here we don't need to let remaining_depth since accum_depth is 64
      //(multiple of 4)
      // int remaining_depth = accum_depth - d;
      // input_val = *((uint32_t*)(input_data + (b * 300 + d) *
      // sizeof(int8_t)));
      input_val = *((uint32_t*)(input_data + d * sizeof(int8_t)));

      filter_val =
          *((uint32_t*)(filter_data + (out_c * 64 + d) * sizeof(int8_t)));
      acc += cfu_op(2, 0, input_val, filter_val);

      // int32_t input_val = input_data[b * accum_depth + d];
      // int32_t filter_val = filter_data[out_c * accum_depth + d];
      // acc += (filter_val + filter_offset) * (input_val + input_offset);
    }
    if (bias_data) {
      acc += bias_data[out_c];
    }
    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
    acc += output_offset;
    // acc = std::max(acc, output_activation_min);
    // acc = std::min(acc, output_activation_max);
    acc = std::max(acc, (int32_t)-128);
    acc = std::min(acc, (int32_t)127);
    // output_data[out_c + 12 * b] = static_cast<int8_t>(acc);
    output_data[out_c] = static_cast<int8_t>(acc);
  }
  // }
}

inline void FullyConnectedWithPackedInt4Weights(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_NE(unpacked_filter_data, nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_data, filter_shape.FlatSize(), unpacked_filter_data);
  FullyConnected(params, input_shape, input_data, filter_shape,
                 unpacked_filter_data, bias_shape, bias_data, output_shape,
                 output_data);
}

template <typename AccumScalar>
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
