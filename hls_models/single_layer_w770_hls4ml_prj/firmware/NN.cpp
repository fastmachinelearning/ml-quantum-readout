#include <iostream>

#include "NN.h"
#include "parameters.h"

void NN(
    input_t fc1_input[N_INPUT_1_1],
    result_t layer4_out[N_LAYER_2]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=fc1_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=fc1_input,layer4_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 3080>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 2>(b2, "b2.txt");
        nnet::load_weights_from_txt<batch_normalization_scale_t, 2>(s4, "s4.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 2>(b4, "b4.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(fc1_input, layer2_out, w2, b2); // fc1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "fc1", N_LAYER_2);
#endif

    nnet::normalize<layer2_t, result_t, config4>(layer2_out, layer4_out, s4, b4); // batch_normalization
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer4_out, "batch_normalization", N_LAYER_2);
#endif

}
