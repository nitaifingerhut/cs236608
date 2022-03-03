#!/bin/bash

cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python main.py --env-type dynamic --exp-repeats 10 --env-topic-change 0,1,2 --recommender autorec_w_topics --steps 500 --num-users 500 --num-items 500 --num-topics 10 --res-dir topics_dyn_autorec_w_topics
python main.py --env-type dynamic-reverse --exp-repeats 10 --env-topic-change 0,1,2 --recommender autorec_w_topics --steps 500 --num-users 500 --num-items 500 --num-topics 10  --res-dir topics_rdyn_autorec_w_topics
python main.py --env-type dynamic --exp-repeats 10 --env-topic-change 0,1,2 --recommender autorec --steps 500 --num-users 500 --num-items 500 --num-topics 10 --res-dir topics_dyn_autorec
python main.py --env-type dynamic-reverse --exp-repeats 10 --env-topic-change 0,1,2 --recommender autorec --steps 500 --num-users 500 --num-items 500 --num-topics 10 --res-dir topics_rdyn_autorec