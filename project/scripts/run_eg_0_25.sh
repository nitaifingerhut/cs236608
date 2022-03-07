#!/bin/bash

cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python -m main --env-type dynamic                           --rec-eps-greedy 0.25 --res-dir env_d_eg_0_25
python -m main --env-type dynamic-reverse                   --rec-eps-greedy 0.25 --res-dir env_dr_eg_0_25
python -m main --env-type dynamic-rec-agnostic              --rec-eps-greedy 0.25 --res-dir env_dra_eg_0_25
python -m main --env-type dynamic-rec-agnostic-rand         --rec-eps-greedy 0.25 --res-dir env_drar_eg_0_25
python -m main --env-type dynamic-reverse-rec-agnostic      --rec-eps-greedy 0.25 --res-dir env_drra_eg_0_25
python -m main --env-type dynamic-reverse-rec-agnostic-rand --rec-eps-greedy 0.25 --res-dir env_drrar_eg_0_25