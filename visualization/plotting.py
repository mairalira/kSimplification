import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Tuple, Union
from typing_extensions import TypedDict
import numpy as np
import matplotlib.colors as mcolors

ColorType = Union[Optional[str]], Tuple[float, float, float, float]


class ScatterParams:
    x_values: List[int]
    y_values: List[float]
    color: ColorType
    marker: Optional[str]

    def __init__(self, x_values: List[int], y_values: List[float], color: ColorType = None,
                 marker: Optional[str] = None):
        self.x_values = x_values
        self.y_values = y_values
        self.color = color
        self.marker = marker


class TSParam:
    x_values: List[int] | np.ndarray
    y_values: List[float] | np.ndarray
    fmat: Optional[str]
    linestyle: Optional[str]
    linewidth: Optional[float]
    color: ColorType

    def __init__(self, x_values: List[int] | np.ndarray, y_values: List[float] | np.ndarray, fmat: str = None,
                 linestyle: Optional[str] = None, linewidth: Optional[float] = None,
                 color: ColorType = None):
        self.x_values = x_values
        self.y_values = y_values
        self.fmat = fmat
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.color = color


class PlotParams:
    ts_params: [TSParam]
    scatter_params: [ScatterParams]
    title: Optional[str]
    save_file: Optional[str]
    display: bool
    x_lim: Optional[Tuple[int, int]]
    y_lim: Optional[Tuple[float, float]]

    def __init__(self, ts_params: List[TSParam] = None, scatter_params: List[ScatterParams] = None,
                 title: Optional[str] = None, save_file: Optional[str] = None,
                 display: bool = False,
                 x_lim: Optional[Tuple[int, int]] = None, y_lim: Optional[Tuple[float, float]] = None):
        if ts_params is None:
            ts_params = []
        if scatter_params is None:
            scatter_params = []

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
        self.scatter_params = scatter_params
        self.title = title
        self.save_file = save_file
        self.display = display
        self.x_lim = x_lim
        self.y_lim = y_lim

    def make_plot(self):
        make_plot(plotParam=self)


def get_args_and_kwargs(params: TSParam | ScatterParams):
    plot_args = [params.x_values, params.y_values]
    plot_kwargs = {}
    if params.color is not None:
        plot_kwargs['color'] = params.color

    if isinstance(params, TSParam):
        if params.fmat is not None:
            plot_args.append(params.fmat)
        if params.linestyle is not None:
            plot_kwargs['linestyle'] = params.linestyle
        if params.linewidth is not None:
            plot_kwargs['linewidth'] = params.linewidth

    if isinstance(params, ScatterParams):
        if params.marker is not None:
            plot_kwargs['marker'] = params.marker
    return plot_args, plot_kwargs


def make_plot(plotParam: PlotParams):
    # Always clear
    plt.clf()
    plt.xlim(plotParam.x_lim)
    plt.ylim(plotParam.y_lim)
    for tsParam in plotParam.ts_params:
        args, kwargs = get_args_and_kwargs(tsParam)
        plt.plot(*args, **kwargs)
    print("Start to scatter on plot")
    for scatterParam in plotParam.scatter_params:
        args, kwargs = get_args_and_kwargs(scatterParam)
        plt.scatter(*args, **kwargs)
    print("Scatter on plot finished")

    if plotParam.title is not None:
        plt.title(plotParam.title)

    if plotParam.save_file is not None:
        plt.savefig(plotParam.save_file)

    if plotParam.display:
        plt.show()


def run():
    color = 'red'
    color = mcolors.to_rgba(color, alpha=0.9)
    fmat = None  # "--"
    linestyle = (0, (2, 0.5))
    linewidth = 5
    x_value = [1, 2]
    y_value = [4, 2]

    ts_param = TSParam(x_values=x_value, y_values=y_value, fmat=fmat, color=color, linestyle=linestyle,
                       linewidth=linewidth)

    color = 'blue'
    marker = "*"
    x_value = [3, 4]
    y_value = [5, 3]
    sc_param = ScatterParams(x_values=x_value, y_values=y_value, color=color, marker=marker)
    location = "testImg"
    x_lim = (0, 6)
    y_lim = (0, 6)
    plot_params = PlotParams(ts_params=[ts_param], scatter_params=[sc_param], save_file=location, display=True,
                             x_lim=x_lim, y_lim=y_lim)

    make_plot(plot_params)


if __name__ == "__main__":
    run()
