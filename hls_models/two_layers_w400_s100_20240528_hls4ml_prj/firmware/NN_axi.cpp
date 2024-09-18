#include "NN_axi.h"

#include <ap_utils.h>

void NN_axi(input_axi_t &in, output_axi_t out[BUFFER_SIZE], bool trigger, unsigned *window_size, unsigned *window_offset, unsigned *scaling_factor, unsigned *out_reset, unsigned *out_offset) {
    //#pragma HLS LATENCY min=1

    // Unregistered axis
	#pragma HLS INTERFACE axis off port=in
	// Registered axis
	//#pragma HLS INTERFACE axis register both port=in
    #pragma HLS INTERFACE bram depth=4294967295 latency=1 port=out
    #pragma HLS INTERFACE ap_none port=trigger
	#pragma HLS INTERFACE s_axilite register port=window_size bundle=config
	#pragma HLS INTERFACE s_axilite register port=window_offset bundle=config
    #pragma HLS INTERFACE s_axilite register port=scaling_factor bundle=config
	#pragma HLS INTERFACE s_axilite register port=out_reset bundle=config
    #pragma HLS INTERFACE s_axilite register port=out_offset bundle=config
    #pragma HLS INTERFACE ap_ctrl_none port=return


	// I/O buffers of the hls4ml NN module
    input_t in_local[N_IQ_WINDOW_IN*2];
    result_t out_local[N_OUT];

    // Index of the output buffer over AXI-lite / MMIO
    unsigned k = 0;

    // Always active
    FOREVER_L: do {

    	// Reset output buffer over AXI-lite / MMIO
        OUT_RESET_C: if ((*out_reset) == 255) {
            k = 0;
            *out_offset = 0;
            RESET_IN_LOCAL_L: for (unsigned i = 0; i < N_IQ_WINDOW_IN*2; i++) {
    			#pragma HLS UNROLL
                in_local[i] = 0;
            }
            RESET_OUT_LOCAL_L: for (unsigned i = 0; i < N_OUT; i++) {
                #pragma HLS UNROLL
                out_local[i] = 0;
            }
            //RESET_OUT_L: for (unsigned i = 0; i < BUFFER_SIZE; i++) {
			//	#pragma HLS UNROLL
            //	out[i] = 0;
            //}
        }

        // Trigger for readout data
        TRIGGER_C: if (trigger) {

        	// If you need you can wait extra clock cycles
            WINDOW_OFFSET_L: ap_wait_n(*window_offset);

            // Read readout data
            LOAD_L: for(unsigned i = 0, j = 0; i < N_IQ_WINDOW_IN; i++) {
                #pragma HLS PIPELINE
            	ap_uint<32> data_in;
            	in.read(data_in);
            	in_local[j++] = data_in.range(15,0) * *scaling_factor;
            	in_local[j++] = data_in.range(31,16) * *scaling_factor;
            }

            // hls4ml NN module
        	NN(in_local, out_local);

        	// Output logits (ground [0], excited [1])
        	STORE_L: for(unsigned i = 0; i < N_OUT; i++){
            	#pragma HLS PIPELINE
            	out[k*2 + i] = out_local[i];
        	}

        	// Increment and reset index of the output buffer over AXI-lite / MMIO
        	k++;
            if (k*2 >= BUFFER_SIZE)
                k = 0;

            // Keep track of the current index via AXI-lite / MMIO
            *out_offset = k;
        }
    } while (true);
}



