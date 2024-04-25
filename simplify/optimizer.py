import numpy as np
from scipy.optimize import minimize


def find_best_solution(X,Y):
    Y = np.array(Y)
    y1 = Y[0]
    xn = X[len(X) - 1]
    x1 = X[0]



    def objective(yp):
        # Line parameters
        a = (yp - y1) / (xn - x1)
        b = y1 - a * x1

        # Predicted y values
        y_pred = a * X + b

        # Squared error
        return np.sum((Y - y_pred) ** 2)
    yp_initial = np.mean(Y)

    # Optimize y'
    result = minimize(objective, yp_initial)

    # Optimal y'
    yp_optimal = result.x[0]
    return yp_optimal

if __name__ == '__main__':
    import random
    random.seed(0)
    X = list(range(5))
    print(X)
    Y = [50,25,-10,10,-30]
    print(Y)
    result = find_best_solution(X,Y)
    print(result)


