#!/bin/bash

sbatch -c 2 -o run_eg_0_00.out -J r00 run_eg_0_00.sh
sbatch -c 2 -o run_eg_0_15.out -J r01 run_eg_0_15.sh
sbatch -c 2 -o run_eg_0_30.out -J r02 run_eg_0_30.sh