#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10 -DIS_BSP=1
make fpga_emu
#make fpga_sim
make report
make fpga
