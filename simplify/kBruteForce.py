
import itertools
import math

def calculate_line_equation(x1, y1, x2, y2, x3):
    # Calculate the slope (m)
    delta_x = x2 - x1
    if delta_x == 0:
        raise ValueError("The points must have different x-coordinates to calculate the slope.")
    m = (y2 - y1) / delta_x

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y3 at x3
    y3 = m * x3 + b

    return y3
def makeTimeSeriesFromPoints(points, length):
    """
    Given a list (points) of [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] of selected points calculate the y value of
    each timeStep.

    For each x in range(timeStep) we have 3 cases:
    1. x1 <= x <= x4: Find the pair xi <= x <=xi+1, s.t. i<=3. Use this slope to find the corresponding y value.
    2. x < x1. Extend the slope between x1 and x2 to x, and find the corresponding y value.
    3. x4 < x. Extend the slope between x3 and x4 to x, and find the corresponding y value.
    :param length: Length of time series
    :return:
    """

    minX = points[0][0]
    maxX = points[-1][0]
    timeSeries = [0 for _ in range(length)]
    pointsX = 0
    for i, x in enumerate(range(length)):
        if pointsX < len(points) - 2 and x > points[pointsX + 1][0]:
            pointsX += 1

        x1 = points[pointsX][0]
        x2 = points[pointsX + 1][0]
        y1 = points[pointsX][1]
        y2 = points[pointsX + 1][1]
        x3 = x
        y3 = calculate_line_equation(x1,y1,x2,y2,x3)
        timeSeries[x] = y3

    return timeSeries


def brute_force(k, time_series):
    """
    Select k timesteps from timeSeries, such that the new time series draw from these points, are as close as possible
    the timeSeries. We use euclidane distance.
    Also the classification should be the same in both.
    :param k:
    :param time_series:
    :return:
    """

    all_x = list(range(len(time_series)))

    min_dist = float('inf')
    best_simplification = None
    choice_of_points = None
    for comb in itertools.combinations(all_x, 3):
        points = [(x, time_series[x]) for x in comb]
        simp_time_series = makeTimeSeriesFromPoints(points, len(time_series))
        dist_to_org = math.dist(simp_time_series, time_series)
        if dist_to_org <= min_dist:
            min_dist = dist_to_org
            choice_of_points = comb
            best_simplification = simp_time_series

    return best_simplification, choice_of_points


if __name__ == '__main__':
    time_series = [1,3,0,6,5,10,5,15,20,25,30]
    print(brute_force(3, time_series))



