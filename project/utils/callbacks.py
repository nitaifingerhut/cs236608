import re

import numpy as np
from loguru import logger


def ARRI_func(env, recommender, **kwargs):
    all_users = env.users
    recommendations, _ = recommender.recommend(all_users, 1)
    true_ratings = env.dense_ratings
    return np.mean(true_ratings[np.arange(len(all_users)), np.squeeze(recommendations)])


def RMSE_func(env, recommender, **kwargs):
    true_ratings = env.dense_ratings
    predicted_ratings = recommender.dense_predictions
    return np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))


def user_preferences_func(env, recommender, user_id: int = 0, **kwargs):
    return env._user_preferences[user_id]


def REC_LAST_STEP_LOSSES_func(env, recommender, **kwargs):
    if env._timestep == kwargs['steps'] - 1:
        return recommender.losses
    else:
        return None


def REC_LAST_STEP_LOSSES_post_proc(L, **kwargs):
    L = L[-1]
    return L


def DO_NOTHING_post_proc(L, **kwargs):
    return L


def eval_post_proc(c):
    try:
        return eval(re.findall("^(.*)_func$", c.__name__)[0] + '_post_proc')
    except NameError as ne:
        logger.info(ne)
        return DO_NOTHING_post_proc
