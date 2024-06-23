from visualization.plotting import TSParam
from typing import List, Optional
import numpy as np
from matplotlib.colors import to_rgba

from utils.model import class_to_color
from models.loadModel import model_classify


def get_ts_param_org(y_org: List[float] | np.ndarray, model_name: str, x_org: List[int] = None,
                     fmat: str = None, linestyle=None, linewidth: float = None) -> TSParam:
    if x_org is None:
        x_org = [i for i, y in enumerate(y_org)]
    if fmat is None:
        fmat = '-'
    if linestyle is None:
        linestyle = "-"
    if linewidth is None:
        linewidth = 2
    # pred_class = model_classify(model_name=model_name, time_series=y_org)
    color = "black"  # class_to_color(pred_class)
    alpha = 1
    color = to_rgba(color, alpha=alpha)
    linewidth = 2

    return TSParam(x_values=x_org, y_values=y_org, fmat=fmat, color=color, linestyle=linestyle, linewidth=linewidth)


def get_ts_param_approx(y_approx: List[float] | np.ndarray, model_name: str, x_org: List[int] = None,
                        linestyle: Optional[str] = None, linewidth: Optional[float] = None) -> TSParam:
    if x_org is None:
        x_org = [i for i, y in enumerate(y_approx)]
    if linestyle is None:
        linestyle = (0, (2, 0.5))
    if linewidth is None:
        linewidth = 5

    pred_class = model_classify(model_name=model_name, time_series=y_approx)
    color = class_to_color(pred_class)
    alpha = 1
    color = to_rgba(color, alpha=alpha)

    return TSParam(x_values=x_org, y_values=y_approx, linestyle=linestyle, linewidth=linewidth, color=color)
