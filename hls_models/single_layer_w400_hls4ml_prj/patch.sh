pwd
FILES=`find . -name myproject_axi_ap_fpext_0_no_dsp_32.vhd`
for f in $FILES; do
	sed -i 's/v7_1_10/v7_1_11/g' $f
	sed -i 's/-- IP Revision: 10/-- IP Revision: 11, patched/g' $f
	sed -i 's/x_ipCoreRevision=10/x_ipCoreRevision=11/g' $f
    echo "Update: $f"
done
if [ -z $FILES ]; then
    echo "Do nothing"
fi
FILES=`find . -name myproject_axi_ap_fpext_2_no_dsp_32.vhd`
for f in $FILES; do
	sed -i 's/v7_1_10/v7_1_11/g' $f
	sed -i 's/-- IP Revision: 10/-- IP Revision: 11, patched/g' $f
	sed -i 's/x_ipCoreRevision=10/x_ipCoreRevision=11/g' $f
    echo "Update: $f"
done
if [ -z $FILES ]; then
    echo "Do nothing"
fi

#sed -i '/module myproject/i (* dont_touch = "yes" *)' hdl/verilog/myproject.v
#------------------------------------------------------------------------
mkdir temp_for_zip_extract
unzip xilinx_com_hls_NN_axi_1_0.zip -d temp_for_zip_extract
cd temp_for_zip_extract
pwd

FILES=`find . -name myproject_axi_ap_fpext_0_no_dsp_32.vhd`
for f in $FILES; do
	sed -i 's/v7_1_10/v7_1_11/g' $f
	sed -i 's/-- IP Revision: 10/-- IP Revision: 11, patched/g' $f
	sed -i 's/x_ipCoreRevision=10/x_ipCoreRevision=11/g' $f
    echo "Update: $f"
done
if [ -z $FILES ]; then
    echo "Do nothing"
fi
FILES=`find . -name myproject_axi_ap_fpext_2_no_dsp_32.vhd`
for f in $FILES; do
	sed -i 's/v7_1_10/v7_1_11/g' $f
	sed -i 's/-- IP Revision: 10/-- IP Revision: 11, patched/g' $f
	sed -i 's/x_ipCoreRevision=10/x_ipCoreRevision=11/g' $f
    echo "Update: $f"
done
if [ -z $FILES ]; then
    echo "Do nothing"
fi

#sed -i '/module myproject/i (* dont_touch = "yes" *)' hdl/verilog/myproject.v

zip -r xilinx_com_hls_NN_axi_1_0.zip .
mv xilinx_com_hls_NN_axi_1_0.zip ..
cd -
rm -rf temp_for_zip_extract
