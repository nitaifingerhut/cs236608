#!/bin/bash

cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python -m main --steps 500 --recommender temporal_autorec --rats-init-mode zeros   --recs-init-mode zeros    --temporal-window-size 1 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_0_rec_0_tw_1_eg_0_0
python -m main --steps 500 --recommender temporal_autorec --rats-init-mode zeros   --recs-init-mode zeros    --temporal-window-size 3 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_0_rec_0_tw_3_eg_0_0
python -m main --steps 500 --recommender temporal_autorec --rats-init-mode zeros   --recs-init-mode zeros    --temporal-window-size 5 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_0_rec_0_tw_5_eg_0_0
python -m main --steps 500 --recommender temporal_autorec --rats-init-mode randint --recs-init-mode randint  --temporal-window-size 1 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_r_rec_r_tw_1_eg_0_0
python -m main --steps 500 --recommender temporal_autorec --rats-init-mode randint --recs-init-mode randint  --temporal-window-size 3 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_r_rec_r_tw_3_eg_0_0
python -m main --steps 500 --recommender temporal_autorec --rats-init-mode randint --recs-init-mode randint  --temporal-window-size 5 --rec-eps-greedy 0.0 --res-dir st_500_ta_rat_r_rec_r_tw_5_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode zeros   --recs-init-mode zeros   --temporal-window-size 1 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_0_rec_0_tw_1_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode zeros   --recs-init-mode zeros   --temporal-window-size 3 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_0_rec_0_tw_3_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode zeros   --recs-init-mode zeros   --temporal-window-size 5 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_0_rec_0_tw_5_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode randint --recs-init-mode randint --temporal-window-size 1 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_r_rec_r_tw_1_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode randint --recs-init-mode randint --temporal-window-size 3 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_r_rec_r_tw_3_eg_0_0
python -m main --steps 500 --recommender temporal_autorec2 --rats-init-mode randint --recs-init-mode randint --temporal-window-size 5 --rec-eps-greedy 0.0 --res-dir st_500_ta2_rat_r_rec_r_tw_5_eg_0_0