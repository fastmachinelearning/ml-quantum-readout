#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#if 0
#include "firmware/NN.h"
#else
#include "firmware/NN_axi.h"
#endif
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 1

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            input_qick_t layer_in;
            for (std::vector<float>::iterator it = in.begin(); it != in.end();) {
            	ap_uint<16> lo = (int) *(it++);
            	ap_uint<16> hi = (int) *(it++);
            	ap_uint<32> word;
            	word.range(15, 0) = lo;
            	word.range(31, 16) = hi;
            	layer_in.write(word);
            }

            output_qick_t layer_out[N_LAYER_5];

            bool trigger = true;
            unsigned window_size = 400;
            unsigned window_offset = 100;
            unsigned scaling_factor = 1;
            unsigned out_reset;
            unsigned out_offset;
            unsigned trigger_delay;

            NN_axi(
            		layer_in,
					layer_out,
					trigger,
					&window_size,
					&window_offset,
					&scaling_factor,
					&out_reset,
					&out_offset
					);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions:           ";
                for(int i = 0; i < N_LAYER_5; i++) {
                  std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions: ";

                nnet::print_result<output_qick_t, N_LAYER_5>(layer_out, std::cout, true);
            }
            e++;

            nnet::print_result<output_qick_t, N_LAYER_5>(layer_out, fout);

        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        input_qick_t layer_in;
        for (unsigned i = 0; i < N_INPUT_1_1; i++) {
        	ap_uint<16> lo = 0;
        	ap_uint<16> hi = 0;
        	ap_uint<32> word;
        	word.range(15, 0) = lo;
        	word.range(31, 16) = hi;
        	layer_in.write(word);
        }

        output_qick_t layer_out[N_LAYER_5];

        bool trigger = true;
        unsigned window_size = 400;
        unsigned window_offset = 100;
        unsigned scaling_factor = 1;
        unsigned out_reset;
        unsigned out_offset;
        unsigned trigger_delay;
        NN_axi(
        		layer_in,
				layer_out,
				trigger,
				&window_size,
				&window_offset,
				&scaling_factor,
				&out_reset,
				&out_offset
				);

        nnet::print_result<output_qick_t, N_LAYER_5>(layer_out, std::cout, false);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
