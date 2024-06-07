from random import random

from Perturbations.dataTypes import PerturbationTS


def create_permutations(points_y, max_y, min_y, k=10 ** 6, e=0.08):
    dataset_dist = abs(max_y - min_y)
    epsilon = e

    change_range = dataset_dist * epsilon
    permutations = []
    for i in range(k):
        new_val_points = []
        for point in points_y:
            random_val = random.gauss(0, 1)  # Currently using gaussion random, could use uniform random.uniform(-1, 1))

            new_val_points.append(point + change_range * random_val)
        permutations.append(new_val_points)

    return permutations


def create_x_y_perturbations(org_pivots_y: List[float], org_pivots_x: List[int], ts_length: int, epsilon: float) -> \
        List[PerturbationTS]:
    resolution = 10 ** 4  # Number of lines
    for i in range(resolution):
        new_pivots_y = []
        new_pivots_x = []
        for j in range(len(org_pivots_x)):
            random_y_change = random.gauss(0, epsilon)
            x_range = 1
            possible_x_values = list(range(-x_range, x_range + 1))
            random_x_change = random.choice(possible_x_values)

            new_pivot_y = org_pivots_y[j] + random_y_change
            new_pivot_x = org_pivots_x[j] + random_x_change
            new_pivots_y.append(new_pivot_y)
            new_pivots_x.append(new_pivot_x)

    return []
