from visualization.plotting import TSParam
from typing import List
import numpy as np
from utils.model import class_to_color
from models.loadModel import model_classify


def get_ts_param_org(y_org: List[float] | np.ndarray, model_name: str, x_org: List[int] = None,
                     fmat: str = None) -> TSParam:
    if x_org is None:
        x_org = [i for i, y in enumerate(y_org)]
    if fmat is None:
        fmat = '-'
    pred_class = model_classify(model_name=model_name, time_series=y_org)
    color = class_to_color(pred_class)
    return TSParam(x_values=x_org, y_values=y_org, fmat=fmat, color=color)


def get_ts_param_approx(y_approx: List[float] | np.ndarray, model_name: str, x_org: List[int] = None,
                        fmat=None) -> TSParam:
    if x_org is None:
        x_org = [i for i, y in enumerate(y_approx)]
    if fmat is None:
        fmat = "--"
    pred_class = model_classify(model_name=model_name, time_series=y_approx)
    color = class_to_color(pred_class)
    return TSParam(x_values=x_org, y_values=y_approx, fmat=fmat, color=color)
