#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cmake .. -DFPGA_DEVICE=intel_a10gx_pac:pac_a10 -DIS_BSP=1
make fpga_emu
make fpga_sim
make report
make fpga
