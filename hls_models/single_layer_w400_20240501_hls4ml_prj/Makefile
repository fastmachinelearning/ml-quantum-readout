# Assuming this repo as a submodule of the qick repo
# qick/submodules/ml-quantum-readout
QICK_BASE_DIR = ../../../../

help:
	@echo "INFO: make <TAB> for targets"
.PHONY: help

hls:
	@vim firmware/NN_axi.cpp +16
	vivado_hls -f build_prj.tcl
	@./profile.sh NN_prj/solution1/syn/report/NN_axi_csynth.xml NN_prj/solution1/impl/report/verilog/NN_axi_export.xml > xilinx_com_hls_NN_axi_1_0.log
.PHONY: hls

patch:
	cp patch.sh NN_prj/solution1/impl/ip && cd NN_prj/solution1/impl/ip && ./patch.sh
.PHONY: patch

update-ip: ultraclean hls patch
	@echo "WARNING: You need to manually copy the IP zip file"
	@echo "WARNING: Match the right name on the target directory"
	@vim firmware/NN_axi.cpp +16
	@echo "WARNING: cp NN_prj/solution1/impl/ip/xilinx_com_hls_NN_axi_1_0.zip ${QICK_BASE_DIR}/qick_ml/ip/<DATE>/xilinx_com_hls_NN_axi_1_0_<CHANGE_HERE>_<FROM>_<TO>.zip"
	@echo "WARNING: cp xilinx_com_hls_NN_axi_1_0.log ${QICK_BASE_DIR}/qick_ml/ip/<DATE>/xilinx_com_hls_NN_axi_1_0_<CHANGE_HERE>_<FROM>_<TO>.log"
.PHONY: update-ip

gui:
	vivado_hls -p NN_prj
.PHONY: gui

clean:
	@echo "INFO: make ultraclean"
.PHONY: clean

ultraclean:
	rm -rf .Xil .hbs *log *rpt *jou NN*_prj tb_data/csim_results.log
.PHONY: ultraclean
