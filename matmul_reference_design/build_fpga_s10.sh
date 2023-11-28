#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cmake .. -DSET_TILE_A=32 -DSET_TILE_B=32 -DFPGA_DEVICE=intel_s10sx_pac:pac_s10 -DIS_BSP=1
make fpga_emu
#make fpga_sim
make report
make fpga
