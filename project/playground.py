import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    with open(Path("~/Desktop/res.pkl").expanduser(), "rb") as f:
        user_preferences = pkl.load(f)["user_preferences"]

    topic_changes = sorted(list(user_preferences.keys()))
    topics = np.linspace(0, 9, num=10, endpoint=True)
    timesteps = np.linspace(1, 99, num=99, endpoint=True)
    xs, ys = np.meshgrid(topics, timesteps)
    for tc in topic_changes:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        zs = np.stack(user_preferences[tc]["mean"], axis=0)
        ax.scatter(xs, ys, zs, cmap="turbo", s=40, c=zs, marker='o', alpha=1)
        ax.set_xlabel('topics')
        ax.set_ylabel('timesteps')
        ax.set_zlabel('preferences')
        ax.set_title(f"topic change = {tc}")
        plt.tight_layout()
        plt.show()

        # zs_o = np.roll(zs, 1, axis=0)
        # print(zs - zs_o)