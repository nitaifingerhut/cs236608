import numpy as np


def ARRI_func(env, recommender, **kwargs):
    all_users = env.users
    recommendations, _ = recommender.recommend(all_users, 1)
    true_ratings = env.dense_ratings
    return np.mean(true_ratings[np.arange(len(all_users)), np.squeeze(recommendations)])


def RMSE_func(env, recommender, **kwargs):
    true_ratings = env.dense_ratings
    predicted_ratings = recommender.dense_predictions
    return np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))


def user_preferences(env, recommender, user_id: int = 0, **kwargs):
    return env._user_preferences[user_id]


def recommender_losses_func(env, recommender, **kwargs):
    return recommender.losses
