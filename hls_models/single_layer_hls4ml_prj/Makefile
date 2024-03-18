help:
	@echo "INFO: make <TAB> for targets"
.PHONY: help

hls:
	vivado_hls -f build_prj.tcl
.PHONY: hls

patch:
	cp patch.sh NN_prj/solution1/impl/ip && cd NN_prj/solution1/impl/ip && ./patch.sh
.PHONY: patch

update-ip: hls
	cp NN_prj/solution1/impl/ip/xilinx_com_hls_NN_axi_1_0.zip ../../../qick_dev/qick_ml
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
