#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cmake ..
make fpga_emu
