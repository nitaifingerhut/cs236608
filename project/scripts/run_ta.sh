#!/bin/bash

cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python -m main --env-type dynamic-reverse              --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 1 --res-dir env_dr_rec_ta_1
python -m main --env-type dynamic-reverse-rec-agnostic --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 1 --res-dir env_drra_rec_ta_1
python -m main --env-type dynamic-reverse              --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 3 --res-dir env_dr_rec_ta_3
python -m main --env-type dynamic-reverse-rec-agnostic --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 3 --res-dir env_drra_rec_ta_3
python -m main --env-type dynamic-reverse              --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 5 --res-dir env_dr_rec_ta_5
python -m main --env-type dynamic-reverse-rec-agnostic --exp-repeats 10 --steps 1000 --recommender temporal_autorec --temporal-window-size 5 --res-dir env_drra_rec_ta_5