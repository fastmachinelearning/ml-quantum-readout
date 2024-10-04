#include <iostream>

#include "NN.h"
#include "parameters.h"

void NN(
    input_t fc1_input[N_INPUT_1_1],
    result_t layer5_out[N_LAYER_5]
) {
#pragma HLS INLINE

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=fc1_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=fc1_input,layer5_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 11200>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 14>(b2, "b2.txt");
        nnet::load_weights_from_txt<batchnorm1_scale_t, 14>(s4, "s4.txt");
        nnet::load_weights_from_txt<batchnorm1_bias_t, 14>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight5_t, 28>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 2>(b5, "b5.txt");
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

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // fc1_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "fc1_relu", N_LAYER_2);
#endif

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::normalize<layer3_t, layer4_t, config4>(layer3_out, layer4_out, s4, b4); // batchnorm1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "batchnorm1", N_LAYER_2);
#endif

    nnet::dense<layer4_t, result_t, config5>(layer4_out, layer5_out, w5, b5); // fc2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer5_out, "fc2", N_LAYER_5);
#endif

}
