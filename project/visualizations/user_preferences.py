import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", type=str, required=True)
    opts = parser.parse_args()

    res_path = Path(__file__).parent.parent.joinpath(f"results/{opts.res_dir}/results.pkl").expanduser()
    with open(res_path, "rb") as f:
        user_preferences = pkl.load(f)["USER_PREF"]

    topic_changes = sorted(list(user_preferences.keys()))
    n_topic_changes = len(topic_changes)
    topics = np.linspace(0, 9, num=10, endpoint=True)
    timesteps = np.linspace(1, 99, num=99, endpoint=True)
    xs, ys = np.meshgrid(topics, timesteps)
    xs, ys = xs.ravel(), ys.ravel()
    dx, dy = 0.5, 0.5

    fig = plt.figure(figsize=(5 * n_topic_changes, 5))
    cmap = plt.cm.magma
    for i, tc in enumerate(topic_changes):
        ax = fig.add_subplot(1, n_topic_changes, i + 1, projection="3d")
        zs = np.stack(user_preferences[tc]["mean"], axis=0).ravel()
        ax.bar3d(
            x=xs, y=ys, z=np.zeros_like(zs), dx=dx, dy=dy, dz=zs, shade=True, color=cmap(plt.Normalize()(zs)),
        )
        ax.set_xlabel("topics")
        ax.set_ylabel("timesteps")
        ax.set_zlabel("preferences")
        ax.set_title(f"topic change = {tc}")

    plt.tight_layout()
    plt.show()
