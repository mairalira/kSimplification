from visualization.plotting import EllipseParams


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
