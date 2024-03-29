import numpy as np

from reclab.environments.environment import Environment
from reclab.environments.topics import Topics
from reclab.recommenders.autorec_w_topics import Autorec_W_Topics
from reclab.recommenders.recommender import Recommender
from reclab.recommenders.libfm import LibFM_MLHB
from tqdm import tqdm
from typing import Callable, Dict, List
from utils.misc import stdout_redirector


def run_simulation(
    env: Environment,
    recommender: Recommender,
    steps: int,
    rpu: int = 1,
    retrain: bool = True,
    callbacks: List[Callable] = [],
    callbacks_kwargs: Dict = dict(),
    post_proc_callbacks: List[Callable] = [],
    post_proc_callbacks_kwargs: Dict = dict(),
    reset: bool = True,
    seed: int = 42,
    label: str = None,
):
    if reset:
        env.seed(seed)
        if hasattr(env, "_topic_change"):
            users, items, ratings = env.reset()
        elif hasattr(env, "_affinity_change "):
            temp = env._affinity_change  # Specificaly for latent factor.
            env._affinity_change = 0
            # items, users, ratings = env.reset()  # Wrong order of return values (manorz, Jan 01)
            users, items, ratings = env.reset()
            env._affinity_change = temp
        else:
            # items, users, ratings = env.reset()
            users, items, ratings = env.reset()
        if isinstance(recommender, Autorec_W_Topics):
            ratings = {k: (v[0], v[1], env._item_topics[k[1]]) for k, v in ratings.items()}
            recommender.set_item_topics(env._item_topics)
        recommender.reset(users, items, ratings)

    results = None
    if len(callbacks) != 0:
        results = [[] for _ in callbacks]

    for i in tqdm(range(steps), desc=label):
        if i != 0:
            for j, callback in enumerate(callbacks):
                res = callback(env, recommender, **callbacks_kwargs)
                results[j].append(res)

        online_users = env.online_users
        recommendations, _ = recommender.recommend(online_users, rpu)
        # print(env._user_preferences[42])
        _, _, ratings, _ = env.step(recommendations)
        # print(env._user_preferences[42])
        # print()
        if isinstance(recommender, Autorec_W_Topics):
            ratings = {k: (v[0], v[1], env._item_topics[k[1]]) for k, v in ratings.items()}
            recommender.set_item_topics(env._item_topics)
        recommender.update(ratings=ratings)

        if retrain and hasattr(recommender, "_model") and isinstance(recommender, LibFM_MLHB):
            with stdout_redirector():
                recommender._model.train(recommender._train_data)  # Specificaly for libfm.
    for j, post_proc_callback in enumerate(post_proc_callbacks):
        results[j] = post_proc_callback(results[j], **post_proc_callbacks_kwargs)
    return results


def run_experiment(
    env_params: Dict,
    recommender: Recommender,
    steps: int,
    repeats: int,
    rpu: int = 1,
    retrain: bool = True,
    callbacks: List[Callable] = [],
    callbacks_kwargs: Dict = dict(),
    post_proc_callbacks: List[Callable] = [],
    post_proc_callbacks_kwargs: Dict = dict(),
    reset: bool = True,
    **kwargs,
):
    callbacks_names = [c.__name__ for c in callbacks]
    callbacks_names = [c[:-5] for c in callbacks_names if c.endswith("_func")]

    if len(kwargs) != 0:
        k = list(kwargs.keys())[0]
        v = list(kwargs.values())[0]

    callbacks_res = {n: {vv: [] for vv in v} for n in callbacks_names}

    if len(kwargs) != 0:
        for vv in v:
            env_params[k] = vv
            env = Topics(**env_params)
            for r in range(repeats):
                _res = run_simulation(
                    env=env,
                    recommender=recommender,
                    steps=steps,
                    rpu=rpu,
                    retrain=retrain,
                    callbacks=callbacks,
                    callbacks_kwargs=callbacks_kwargs,
                    post_proc_callbacks=post_proc_callbacks,
                    post_proc_callbacks_kwargs=post_proc_callbacks_kwargs,
                    reset=reset,
                    seed=r,
                    label=f"{k}={vv}, repeat={str(r).zfill(2)}",
                )
                for i, n in enumerate(callbacks_names):
                    callbacks_res[n][vv].append(_res[i])

            for i, n in enumerate(callbacks_names):
                # callbacks_res[n][vv] = list(np.mean(np.asarray(callbacks_res[n][vv]), axis=0))
                callbacks_res[n][vv] = {
                    "mean": list(np.mean(np.asarray(callbacks_res[n][vv]), axis=0)),
                    "std": list(np.std(np.asarray(callbacks_res[n][vv]), axis=0)),
                }

    return callbacks_res
