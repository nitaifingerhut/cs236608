from asyncio.log import logger
from lib2to3.pgen2.pgen import DFAState
from ntpath import join
import os
import argparse
import imageio
import matplotlib.pyplot as plt
import math
import numpy as np

from reclab.environments.topics import Topics
from reclab.recommenders import RECOMMENDERS
from utils.callbacks import ARRI_func, RMSE_func, user_preferences
from utils.plots import plot_graphs, plt_to_numpy
from utils.simulation import run_simulation, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--recommender", type=str, default="autorec", choices=RECOMMENDERS.keys())
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--num-items", type=int, default=100)
    parser.add_argument("--num-topics", type=int, default=10)
    parser.add_argument("--rating-freq", type=float, default=0.2)
    parser.add_argument("--res-dir", type=str)
    parser.add_argument("--env-type", type=str, choices=('dynamic', 'dynamic-reverse'), default='dynamic')
    parser.add_argument("--exp-repeats", type=int, default=1)
    # parser.add_argument("--env-topic-change", nargs='+', type=float)  # For some weird bug in nargs='+' only in VsCode... (manorz, 02/12/22)
    parser.add_argument("--env-topic-change", type=str, default='0,1,2')
    parser.add_argument("--rec-eps-greedy", type=float)
    parser.add_argument("--recommender_mode", type=str, default='baseline')
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    try:
        opts.res_dir = os.path.join('results', opts.res_dir)
    except TypeError as _:
        opts.res_dir = os.path.join('results', opts.env_type, opts.recommender)
        if opts.rec_eps_greedy:
            opts.res_dir = os.path.join(opts.res_dir, f'eps_greedy={opts.rec_eps_greedy}')
    opts.res_dir = os.path.abspath(opts.res_dir)
    try:
        os.makedirs(opts.res_dir)
    except FileExistsError as e:
        logger.warning(e)
        os.makedirs(opts.res_dir, exist_ok=True)
    if opts.rec_eps_greedy:
        opts.rec_eps_greedy = {'type': 'eps_greedy', 'eps': opts.rec_eps_greedy}
    
    opts.env_topic_change = [float(x) for x in opts.env_topic_change.split(',')]

    env_params = {
        "num_topics": opts.num_topics,
        "num_users": opts.num_users,
        "num_items": opts.num_items,
        "rating_frequency": opts.rating_freq,
        "num_init_ratings": opts.num_users * opts.num_items // 5,
        "noise": 0.0,
        "user_model": opts.env_type,
    }
    
    Autorec_params = {
        "num_users": opts.num_users,
        "num_items": opts.num_items,
        "hidden_neuron": 500,
        "train_epoch": 100,
        "random_seed": opts.seed,
        "recommender_mode": opts.recommender_mode
    }

    recommender = RECOMMENDERS[opts.recommender](**Autorec_params)

    if opts.rec_eps_greedy:
        recommender.update_strategy(opts.rec_eps_greedy)

    callbacks_kwargs = dict(user_id=0)
    callbacks = [RMSE_func, ARRI_func]#, user_preferences]

    repeats = opts.exp_repeats
    topic_changes = opts.env_topic_change
    res = run_experiment(
        env_params=env_params,
        recommender=recommender,
        steps=opts.steps,
        repeats=repeats,
        rpu=1,
        retrain=True,
        callbacks=callbacks,
        callbacks_kwargs=callbacks_kwargs,
        reset=True,
        topic_change=topic_changes,
    )
    for k, v in res.items():
        plot_graphs(*list(v.values()), title=k, legend=True, labels=[f"topic_change={x}" for x in topic_changes], save=os.path.join(opts.res_dir, k))
