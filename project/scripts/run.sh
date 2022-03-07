#!/bin/bash

sbatch -c 2 -o run_eg_0_00.out -J r00 run_eg_0_00.sh
sbatch -c 2 -o run_eg_0_25.out -J r01 run_eg_0_25.sh
sbatch -c 2 -o run_eg_0_50.out -J r02 run_eg_0_50.sh