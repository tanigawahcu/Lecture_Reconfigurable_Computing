# Intel DevCloud for OneAPI 用のbuild&run shell スクリプト
rm -rf build
mkdir build
sub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d build build_fpga_s10.sh
sub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d build run_fpga_s10.sh
