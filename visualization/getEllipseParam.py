import numpy as np

from visualization.plotting import EllipseParams
from Perturbations.dataTypes import SegmentedTS


def make_ellipse_param(x_pos: int, y_pos: float, inner: bool):
    if inner:
        radius = 0.5
        color = "grey"
        fill = True
    else:
        radius = 0.6
        color = "black"
        fill = False

    ellipse_param = EllipseParams(x_pos=x_pos, y_pos=y_pos, radius=radius, color=color, fill=fill)
    return ellipse_param


def make_all_ellipse_param(x_pivots, y_pivots, inner: bool):
    ellipse_params = []
    for x_pivot, y_pivot in zip(x_pivots, y_pivots):
        ellipse_param = make_ellipse_param(x_pivot, y_pivot, inner)
        ellipse_params.append(ellipse_param)
    return ellipse_params


def both_in_and_out_ellipse_params(approximation: SegmentedTS):
    ellipseIn = make_all_ellipse_param(x_pivots=approximation.x_pivots, y_pivots=approximation.y_pivots, inner=True)
    out_x_pivots = np.concatenate([[0], approximation.x_pivots[1:-1], [len(approximation.line_version) - 1]])
    out_y_pivots = np.concatenate(
        [[approximation.line_version[0]], approximation.y_pivots[1:-1], [approximation.line_version[-1]]])
    ellipseOut = make_all_ellipse_param(x_pivots=out_x_pivots, y_pivots=out_y_pivots, inner=False)

    all_ellipse = ellipseIn + ellipseOut
    return all_ellipse


def add_one_more_circle_over_all(approximation: SegmentedTS):
    out_x_pivots = [0] + approximation.x_pivots[1:-1] + [len(approximation.line_version) - 1]
    out_y_pivots = [approximation.line_version[0]] + approximation.y_pivots[1:-1] + [approximation.line_version[-1]]
    x_y_pairs = []
    for x, y in zip(out_x_pivots, out_y_pivots):
        x_y_pairs.append((x, y))
    for x, y in zip(approximation.x_pivots, approximation.y_pivots):
        if (x, y) in x_y_pairs:  # Dont add duplicate
            continue
        x_y_pairs.append((x, y))

    all_outout = []
    for out_x, out_y in x_y_pairs:
        outer_ellipse = EllipseParams(x_pos=out_x, y_pos=out_y, radius=0.7, color="brown", fill=True)
        all_outout.append(outer_ellipse)
    return all_outout
