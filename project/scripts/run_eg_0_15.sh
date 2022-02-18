source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate double-ml
echo "hello from $(python --version) in $(which python)"

python -m main --steps 1000 --temporal-window-size 1  --rating-freq 0.2 --rec-eps-greedy 0.15 --res-dir st_1000_tp_01_rf_0_2_eg_0_15
python -m main --steps 1000 --temporal-window-size 3  --rating-freq 0.2 --rec-eps-greedy 0.15 --res-dir st_1000_tp_03_rf_0_2_eg_0_15
python -m main --steps 1000 --temporal-window-size 5  --rating-freq 0.2 --rec-eps-greedy 0.15 --res-dir st_1000_tp_05_rf_0_2_eg_0_15
python -m main --steps 1000 --temporal-window-size 10 --rating-freq 0.2 --rec-eps-greedy 0.15 --res-dir st_1000_tp_10_rf_0_2_eg_0_15
python -m main --steps 1000 --temporal-window-size 1  --rating-freq 0.5 --rec-eps-greedy 0.15 --res-dir st_1000_tp_01_rf_0_5_eg_0_15
python -m main --steps 1000 --temporal-window-size 3  --rating-freq 0.5 --rec-eps-greedy 0.15 --res-dir st_1000_tp_03_rf_0_5_eg_0_15
python -m main --steps 1000 --temporal-window-size 5  --rating-freq 0.5 --rec-eps-greedy 0.15 --res-dir st_1000_tp_05_rf_0_5_eg_0_15
python -m main --steps 1000 --temporal-window-size 10 --rating-freq 0.5 --rec-eps-greedy 0.15 --res-dir st_1000_tp_10_rf_0_5_eg_0_15
python -m main --steps 1000 --temporal-window-size 1  --rating-freq 0.8 --rec-eps-greedy 0.15 --res-dir st_1000_tp_01_rf_0_8_eg_0_15
python -m main --steps 1000 --temporal-window-size 3  --rating-freq 0.8 --rec-eps-greedy 0.15 --res-dir st_1000_tp_03_rf_0_8_eg_0_15
python -m main --steps 1000 --temporal-window-size 5  --rating-freq 0.8 --rec-eps-greedy 0.15 --res-dir st_1000_tp_05_rf_0_8_eg_0_15
python -m main --steps 1000 --temporal-window-size 10 --rating-freq 0.8 --rec-eps-greedy 0.15 --res-dir st_1000_tp_10_rf_0_8_eg_0_15