#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1520
#define N_LAYER_2 2
#define N_LAYER_2 2

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<14,14> input_t;
typedef ap_fixed<18,18> fc1_accum_t;
typedef ap_fixed<18,18> layer2_t;
typedef ap_fixed<6,1> weight2_t;
typedef ap_fixed<6,1> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<20,3> batch_normalization_scale_t;
typedef ap_fixed<20,3> batch_normalization_bias_t;

#endif
