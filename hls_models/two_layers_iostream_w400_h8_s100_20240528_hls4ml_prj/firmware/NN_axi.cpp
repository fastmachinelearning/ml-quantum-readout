#include "NN_axi.h"

#include <ap_utils.h>

bool load(input_qick_t &in, hls::stream<input_t> &in_local, unsigned scaling_factor) {
	#pragma HLS INLINE off
    input_t ctype;
    #pragma HLS DATA_PACK variable=ctype
    LOAD_L: for(unsigned i = 0, j = 0; i < N_IQ_WINDOW_IN; i++) {
    	ap_uint<32> data_in;
    	in.read(data_in);
        ctype[j++] = (typename input_t::value_type(data_in.range(15,0))) * scaling_factor;
        ctype[j++] = (typename input_t::value_type(data_in.range(31,16))) * scaling_factor;
    }
    in_local.write(ctype);

    return true;
}

void store(hls::stream<result_t> &out_local, int offset, output_qick_t *out) {
	#pragma HLS INLINE off
	result_t ctype = out_local.read();
	#pragma HLS DATA_PACK variable=ctype
	STORE_L: for(unsigned i = 0; i < N_OUT; i++) {
		out[offset + i] = ctype[i].to_float();
	}
}

void NN_axi(
		input_qick_t &in,
		output_qick_t out[BUFFER_SIZE],
		bool trigger,
		unsigned *window_size,
		unsigned *window_offset,
		unsigned *scaling_factor,
		unsigned *out_reset,
		unsigned *out_offset
		) {

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
    hls::stream<input_t> in_local("input_1");
    hls::stream<result_t> out_local("output_1");

    #pragma HLS STREAM variable=in_local depth=1
    #pragma HLS STREAM variable=out_local depth=1

    // Index of the output buffer over AXI-lite / MMIO
    unsigned k = 0;

    // Always active
#ifdef __SYNTHESIS__
    FOREVER_L: do {

    	// Reset output buffer over AXI-lite / MMIO
        //OUT_RESET_C: if ((*out_reset) == 255) {
        //    k = 0;
        //    *out_offset = 0;
        //    RESET_IN_LOCAL_L: for (unsigned i = 0; i < N_IQ_WINDOW_IN*2; i++) {
    	//		#pragma HLS UNROLL
        //        in_local[i] = 0;
        //    }
        //    RESET_OUT_LOCAL_L: for (unsigned i = 0; i < N_OUT; i++) {
        //        #pragma HLS UNROLL
        //        out_local[i] = 0;
        //    }
        //    //RESET_OUT_L: for (unsigned i = 0; i < BUFFER_SIZE; i++) {
		//	//	#pragma HLS UNROLL
        //    //	out[i] = 0;
        //    //}
        //}

        // Trigger for readout data
        TRIGGER_C: if (trigger) {

        	// If you need you can wait extra clock cycles
            WINDOW_OFFSET_L: ap_wait_n(*window_offset);
#endif

            // Read readout data
//            input_t in_ctype;
//            #pragma HLS DATA_PACK variable=in_ctype
//            LOAD_L: for(unsigned i = 0, j = 0; i < N_IQ_WINDOW_IN; i++) {
//            	ap_uint<32> data_in;
//            	in.read(data_in);
//                in_ctype[j++] = (typename input_t::value_type(data_in.range(15,0))) * *scaling_factor;
//                in_ctype[j++] = (typename input_t::value_type(data_in.range(31,16))) * *scaling_factor;
//            }
//            in_local.write(in_ctype);

            bool load_done = load(in, in_local, *scaling_factor);

            // hls4ml NN module
        	NN(in_local, out_local);

        	// Output logits (ground [0], excited [1])

        	if (load_done) {
            result_t out_ctype = out_local.read();
			#pragma HLS DATA_PACK variable=out_ctype
            STORE_L: for(unsigned i = 0; i < N_OUT; i++) {
                out[k*2 + i] = out_ctype[i].to_float();
            }
        	}
//        	store(out_local, k*2, out);

        	// Increment and reset index of the output buffer over AXI-lite / MMIO
        	k++;
            if (k*2 >= BUFFER_SIZE)
                k = 0;

            // Keep track of the current index via AXI-lite / MMIO
            *out_offset = k;
#ifdef __SYNTHESIS__
        }
    } while (true);
#endif
}



