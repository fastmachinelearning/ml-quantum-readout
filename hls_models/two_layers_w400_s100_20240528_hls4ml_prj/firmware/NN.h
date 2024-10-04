#ifndef NN_H_
#define NN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void NN(
    input_t fc1_input[N_INPUT_1_1],
    result_t layer5_out[N_LAYER_5]
);

#endif
