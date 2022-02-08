import argparse
import imageio
import matplotlib.pyplot as plt
import math
import numpy as np

from reclab.environments.topics import Topics
from reclab.recommenders.libfm import LibFM_MLHB
from utils.callbacks import ARRI_func, RMSE_func, user_preferences
from utils.plots import plot_graphs, plt_to_numpy
from utils.simulation import run_simulation, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--num-items", type=int, default=100)
    parser.add_argument("--num-topics", type=int, default=10)
    parser.add_argument("--rating-freq", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()

    env_params = {
        "num_topics": opts.num_topics,
        "num_users": opts.num_users,
        "num_items": opts.num_items,
        "rating_frequency": opts.rating_freq,
        "num_init_ratings": opts.num_users * opts.num_items // 5,
        "noise": 0.0,
    }
    env = Topics(**env_params)
    env.seed(opts.seed)

    LibFM_params = {
        "num_user_features": 0,
        "num_item_features": 0,
        "num_rating_features": 0,
        "max_num_users": opts.num_users,
        "max_num_items": opts.num_items,
        "seed": opts.seed,
    }
    recommender = LibFM_MLHB(**LibFM_params)

    callbacks_kwargs = dict(user_id=0)
    callbacks = [RMSE_func, ARRI_func, user_preferences]

    env = Topics(**env_params, topic_change=1)
    RMSEs, ARRIs, PREFERENCES = run_simulation(
        env=env,
        recommender=recommender,
        steps=opts.steps,
        rpu=1,
        retrain=True,
        callbacks=callbacks,
        callbacks_kwargs=callbacks_kwargs,
        reset=True,
        seed=opts.seed,
    )
    plot_graphs(RMSEs, title="rmse")
    plot_graphs(ARRIs, title="arri")

    vid_writer = imageio.get_writer(
        "preferences.mov", fps=5, macro_block_size=1, codec="prores_ks", pixelformat="yuv444p10le",
    )
    PREFERENCES = np.stack(PREFERENCES, axis=0)
    bins = np.linspace(math.floor(np.min(PREFERENCES)), math.ceil(np.max(PREFERENCES)) + 1, num=10)
    for x in PREFERENCES:
        _, ax = plt.subplots()
        ax.hist(x, bins=bins)
        data = plt_to_numpy(ax)
        plt.close()
        vid_writer.append_data(data)
    vid_writer.close()

    repeats = 5
    topic_changes = [0, 1, 2]
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
        topic_change=topic_changes
    )
    for k, v in res.items():
      plot_graphs(*list(v.values()), title=k, legend=True, labels=[f"topic_change={x}" for x in topic_changes])
