#!/bin/bash

sbatch -c 2 -o run_ta.out -J run00 run_ta.sh
sbatch -c 2 -o run_ta2.out -J run01 run_ta2.sh