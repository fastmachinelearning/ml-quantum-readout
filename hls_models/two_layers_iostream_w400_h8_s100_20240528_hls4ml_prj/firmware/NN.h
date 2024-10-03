#ifndef NN_H_
#define NN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void NN(
    hls::stream<input_t> &fc1_input,
    hls::stream<result_t> &layer5_out
);

#endif
