import matplotlib.pyplot as plt

from typing import Optional, List
from visualization.plotting import PlotParams, get_args_and_kwargs


class MultiPlotParams:
    plotParams: [PlotParams]
    title_over_all: Optional[str]
    folder: Optional[str]
    save_file: Optional[str]
    display: bool
    rows: int
    cols: int

    def __init__(self, plotParams: List[PlotParams] = None, title: Optional[str] = None,
                 save_file: Optional[str] = None, folder: Optional[str] = None,
                 display: bool = False, rows: int = None, cols: int = None):
        if plotParams is None:
            plotParams = []
        if rows is None and cols is None:
            cols = 4
            rows = len(plotParams) // cols
            if cols * rows < len(plotParams):
                rows += 1

        self.plotParams = plotParams
        self.title_over_all = title
        self.save_file = save_file
        self.folder = folder
        self.display = display
        self.rows = rows
        self.cols = cols

    def plot(self):
        make_plot(self)


def make_plot(multiplot: MultiPlotParams):
    # Always clear
    plt.clf()
    golden = (1 + 5 ** 0.5) / 2
    height = 10
    ratio = 3 / 2 * golden
    width = int(height * ratio)
    fig, ax = plt.subplots(multiplot.rows, multiplot.cols, figsize=(width, height))

    if multiplot.rows == 1 or multiplot.cols == 1:
        ax = np.array(ax).reshape(rows, cols)

    for nr, plotParam in enumerate(multiplot.plotParams):
        col = nr % multiplot.cols
        row = nr // multiplot.cols
        curr_ax = ax[row][col]
        curr_ax.set_xlim(plotParam.x_lim)
        curr_ax.set_ylim(plotParam.y_lim)
        if col != 0:
            curr_ax.get_yaxis().set_visible(False)
        for scatterParam in plotParam.scatter_params:
            args, kwargs = get_args_and_kwargs(scatterParam)
            curr_ax.scatter(*args, **kwargs, zorder=1)
        for tsParam in plotParam.ts_params:
            args, kwargs = get_args_and_kwargs(tsParam)
            curr_ax.plot(*args, **kwargs, zorder=2)

        for ellipseParam in plotParam.ellipse_params:
            x_axsis_r = ellipseParam.radius
            y_axsis_r = ellipseParam.radius * (
                    abs(plotParam.y_lim[0] - plotParam.y_lim[1]) / abs(plotParam.x_lim[0] - plotParam.x_lim[1]))
            ellipse = patches.Ellipse((ellipseParam.x_pos, ellipseParam.y_pos), x_axsis_r, y_axsis_r,
                                      fill=ellipseParam.fill, color=ellipseParam.color, zorder=3)
            curr_ax.add_patch(ellipse)

        if plotParam.title is not None:
            curr_ax.title.set_text(plotParam.title)
    # fig.figure(figsize=(10, 10))
    if multiplot.save_file is not None:
        folder = multiplot.folder
        if folder is None:
            folder = "PyPlots"
        plt.savefig(f"{folder}/{multiplot.save_file}.png", dpi=100)

    if multiplot.display:
        plt.show()

    plt.close(fig)
