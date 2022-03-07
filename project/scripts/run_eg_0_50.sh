#!/bin/bash

cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python -m main --env-type dynamic                           --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_d_eg_0_5
python -m main --env-type dynamic-reverse                   --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_dr_eg_0_5
python -m main --env-type dynamic-rec-agnostic              --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_dra_eg_0_5
python -m main --env-type dynamic-rec-agnostic-rand         --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_drar_eg_0_5
python -m main --env-type dynamic-reverse-rec-agnostic      --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_drra_eg_0_5
python -m main --env-type dynamic-reverse-rec-agnostic-rand --exp-repeats 10 --rec-eps-greedy 0.5 --res-dir env_drrar_eg_0_5