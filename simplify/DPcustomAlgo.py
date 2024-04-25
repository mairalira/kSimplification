import numpy as np
from collections import defaultdict as D
from collections import namedtuple as T
from simplify.plotting import plot
from typing import Dict
import random


Solution = T("Solution", "error index prev last_seg")
Function = T("Function", "m b")


def sol(error, index, previous, last_seg):
    return Solution(error, index, previous,last_seg)  # please never round


sol0 = sol(0, 0, 0, True)
VINF = sol(float("inf"), 0, 0, True)


def _line_error(f:Function,X,Y):
    error = 0
    for x, y in zip(X,Y):
        error += abs(f.m*x+f.b-y)**2
    return error


def _gen_line(x1,y1,x2,y2) -> Function:
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return Function(m, b)


def segmented_least_squares(X, Y, c) ->Dict[int,Solution]:
    """
    Least squares solution using segmented least squares.
    Parameters
    @ X:param, List of X values
    @ Y:param, List of Y values
    @ c:param, Punishment for more segments
    """
    OPT = D(lambda: VINF)
    # Base case for only one point
    OPT[0] = sol0

    # Solve DP
    for i in range(1, len(X)):
        min_error = float('inf')
        min_j = None
        last_seg = False
        for j in range(0, i):
            f = _gen_line(X[j],Y[j],X[i],Y[i])

            # Keep other segments
            f_line_error = _line_error(f,X[j:i+1], Y[j:i+1])
            f_segment_error = f_line_error + c
            f_segment_keep_error = OPT[j].error + f_segment_error
            if f_segment_keep_error < min_error:
                min_error = f_segment_keep_error
                min_j = j
                last_seg = False

            # Do not keep other segments
            f_total_line_error = _line_error(f, X[0:i+1], Y[0:i+1])
            f_total_error = f_total_line_error + c
            if f_total_error < min_error:
                min_error = f_total_error
                min_j = j
                last_seg = True

        OPT[i] = sol(min_error, i, min_j, last_seg)

    # Currently OPT[n-1] holds the best solution, if we require the algorithm to pick n, lets fix this.
    # Let's manually check all possible last segments, going through two points in the TS.
    # We will use the DP for the area before last segment.
    min_error = OPT[len(X)-1].error
    for i in range(1, len(X)):

        for j in range(0, i):
            if j == 19 and i == 21:
                print("Stop")
            f = _gen_line(X[j],Y[j],X[i],Y[i])
            f_line_out_error = _line_error(f,X[j:], Y[j:])
            f_out_error = OPT[j].error + c + f_line_out_error # From 0..j + c + rest
            if f_out_error < min_error:
                OPT[len(X)-1] = sol(f_out_error, i, j, False)
                min_error = OPT[len(X)-1].error
    return OPT


def solve(X, Y, c) -> Dict[int,Solution]:
    OPT = segmented_least_squares(X, Y, c)
    return OPT


def extract_points(OPT: Dict[int, Solution], X):
    solution = OPT[len(X)-1]
    list_of_points_last_first = [solution.index]
    while (not solution.last_seg) and (solution.index != solution.prev):
        solution = OPT[solution.prev]
        list_of_points_last_first.append(solution.index)

    if solution.last_seg:
        list_of_points_last_first.append(solution.prev)

    return list(reversed(list_of_points_last_first))


def solve_and_find_points(X,Y,c,saveImg=True):
    OPT = solve(X, Y, c)
    print(OPT[len(X) - 1].error)


    selected_points = extract_points(OPT, X)
    print(f"Num points:{len(selected_points)}, all points: {selected_points}")

    ys = [Y[i] for i in selected_points]
    if saveImg:
        plot(X, Y, selected_points, ys)
    return selected_points, ys

if __name__ == "__main__":
    random.seed(2)
    X = list(range(1000))
    Y = [random.randint(-100,100) for _ in range(len(X))]
    c = 11000
    solve_and_find_points(X,Y,c)
