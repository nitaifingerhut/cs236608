from reclab.environments.environment import Environment
from reclab.recommenders.recommender import Recommender
from tqdm import tqdm
from typing import Callable, Dict, List
from utils.misc import stdout_redirector


def simulation_run(
    env: Environment,
    recommender: Recommender,
    steps: int,
    rpu: int = 1,
    retrain: bool = True,
    callbacks: List[Callable] = [],
    callbacks_kwargs: Dict = dict(),
    reset: bool = True,
    seed: int = 42,
):
    if reset:
        env.seed(seed)
        if hasattr(env, "_topic_change"):
            temp = env._topic_change  # Specificaly for topics.
            env._topic_change = 0
            # items, users, ratings = env.reset()  # Wrong order of return values (manorz, Jan 01)
            users, items, ratings = env.reset()
            env._topic_change = temp
        elif hasattr(env, "_affinity_change "):
            temp = env._affinity_change  # Specificaly for latent factor.
            env._affinity_change = 0
            # items, users, ratings = env.reset()
            users, items, ratings = env.reset()
            env._affinity_change = temp
        else:
            # items, users, ratings = env.reset()
            users, items, ratings = env.reset()

        recommender.reset(items, users, ratings)

    results = None
    if len(callbacks) != 0:
        results = [[] for _ in callbacks]

    for i in tqdm(range(steps)):
        for j, callback in enumerate(callbacks):
            res = callback(env, recommender, **callbacks_kwargs)
            results[j].append(res)

        online_users = env.online_users
        recommendations, predicted_ratings = recommender.recommend(online_users, rpu)
        _, _, ratings, _ = env.step(recommendations)
        recommender.update(ratings=ratings)

        if retrain and hasattr(recommender, "_model"):
            with stdout_redirector():
                recommender._model.train(recommender._train_data)  # Specificaly for libfm.
    return results
