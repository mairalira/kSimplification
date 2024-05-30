import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Tuple
from typing_extensions import TypedDict
import numpy as np


class TSParam:
    x_values: List[int] | np.ndarray
    y_values: List[float] | np.ndarray
    fmat: Optional[str]
    color: Optional[str]

    def __init__(self, x_values: List[int] | np.ndarray, y_values: List[float] | np.ndarray, fmat: str = None,
                 color: Optional[str] = None):
        self.x_values = x_values
        self.y_values = y_values
        self.fmat = fmat
        self.color = color


class PlotParams:
    ts_params: [TSParam]
    title: Optional[str]
    save_file: Optional[str]
    display: bool
    x_lim: Optional[Tuple[int, int]]
    y_lim: Optional[Tuple[int, int]]

    def __init__(self, ts_params: List[TSParam], title: Optional[str] = None, save_file: Optional[str] = None,
                 display: bool = False,
                 x_lim: Optional[Tuple[int, int]] = None, y_lim: Optional[Tuple[int, int]] = None):
        if x_lim is None:
            min_x = min([min([x for x in ts_param.x_values]) for ts_param in ts_params])
            max_x = max([max([x for x in ts_param.x_values]) for ts_param in ts_params])
            x_lim = (min_x, max_x)
            self.x_lim = x_lim
        if y_lim is None:
            min_y = min([min([y for y in ts_param.y_values]) for ts_param in ts_params])
            max_y = max([max([y for y in ts_param.y_values]) for ts_param in ts_params])
            y_lim = (min_y, max_y)
            self.y_lim = y_lim

        self.ts_params = ts_params
        self.title = title
        self.save_file = save_file
        self.display = display
        self.x_lim = x_lim
        self.y_lim = y_lim


def make_plot(plotParam: PlotParams):
    # Always clear
    plt.clf()
    plt.xlim(plotParam.x_lim)
    plt.ylim(plotParam.y_lim)
    for tsParam in plotParam.ts_params:
        if tsParam.fmat is not None and tsParam.color is not None:
            print("here hallo?")
            print(tsParam.x_values, tsParam.y_values, tsParam.fmat)
            plt.plot(tsParam.x_values, tsParam.y_values, tsParam.fmat, color=tsParam.color)
        elif tsParam.fmat is None and tsParam.color is not None:
            plt.plot(tsParam.x_values, tsParam.y_values, color=tsParam.color)
        elif tsParam.fmat is not None and tsParam.color is None:
            plt.plot(tsParam.x_values, tsParam.y_values, tsParam.fmat)
        else:  # tsParam.styles is None and tsParam.color is None:
            plt.plot(tsParam.x_values, tsParam.y_values)

    if plotParam.title is not None:
        plt.title(plotParam.title)
    if plotParam.save_file is not None:
        plt.savefig(plotParam.save_file)

    if plotParam.display:
        plt.show()


def run():
    x_value = [1, 2, 3]
    y_value = [2, 3, 4]
    plt.plot(x_value, y_value)
    plt.show()
    color = 'red'
    fmat = "--"
    location = "testImg"
    x_lim = (0, 4)
    y_lim = (0, 4)

    ts_param = TSParam(x_values=x_value, y_values=y_value, fmat=fmat, color=color)
    plot_params = PlotParams(ts_params=[ts_param], save_file=location, display=True)
    make_plot(plot_params)


if __name__ == "__main__":
    run()
