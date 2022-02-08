import io
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Union


def plot_graphs(
    *lists, title: str, legend: bool = False, labels: List[str] = None, show: bool = True, save: str = None
):
    _, ax = plt.subplots()
    for i, l in enumerate(lists):
        ax.plot(np.arange(len(l)), l, ".-", label=f"{i + 1}" if labels is None else labels[i])
    ax.set_xlabel(r"$timestep$")
    ax.set_title(title)
    if legend:
        ax.legend()

    if save:
        plt.savefig(save + ".png")

    if show:
        plt.show()
    else:
        plt.close()


def plt_to_numpy(fig_or_ax: Union[plt.Axes, plt.Figure]) -> np.ndarray:
    """
    Takes a matplotlib figure/axes and return it as a numpy array in high resolution.
    Note: since it saves to buffer in `raw` mode, it's the fastest implementation possible
    that maintains figure in high resolution.

    Args:
        fig_or_ax: Matplotlib Figure or Axes.

    Returns:
        A numpy array with the figure data, with the shape (fig_width, fig_height, 4).
    """
    if not isinstance(fig_or_ax, plt.Figure) and not isinstance(fig_or_ax, plt.Axes):
        raise TypeError(f"expecting type [plt.Axes, plt.Figure] but `fig_or_ax` has type {type(fig_or_ax)}")

    fig = fig_or_ax.figure if isinstance(fig_or_ax, plt.Axes) else fig_or_ax

    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))
