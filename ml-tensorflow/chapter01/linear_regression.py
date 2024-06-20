import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
learning_rate = 0.01
num_iterations = 1000
num_points = 100

# generate random data
np.random.seed(0)
X = np.random.randn(num_points, 1)
y = 0.5 * X + 2 + np.random.randn(num_points, 1) * 0.5

# plot the data
plt.plot(X, y, 'rx')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


def compute_error_for_line_given_points(b, w, points):
    """
    Computes the error for the line given the points (Model!)
    :param b: y-intercept of the line
    :param w: slope of the line
    :param points: list of points
    :return: error of the line
    """

    total_error = 0
    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        total_error += (y - (w * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    """
    Computes the gradient of the error function with respect to b and w
    :param b_current: current y-intercept of the line
    :param w_current: current slope of the line
    :param points: list of points
    :param learning_rate: learning rate
    :return: updated b and w
    """

    b_gradient, w_gradient = 0, 0
    n = float(len(points))

    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        b_gradient += -(2 / n) * (y - (w_current * x + b_current))  # dE/db = 2(wx+b-y)
        w_gradient += -(2 / n) * x * (y - (w_current * x + b_current))  # dE/dw = 2x(wx+b-y)

    # update the parameters
    b_new = b_current - (learning_rate * b_gradient)
    w_new = w_current - (learning_rate * w_gradient)

    return [b_new, w_new]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """
    Runs the gradient descent algorithm
    :param points: list of points
    :param starting_b: starting y-intercept of the line
    :param starting_w: starting slope of the line
    :param learning_rate: learning rate
    :param num_iterations: number of iterations to run
    :return: updated b and w
    """

    b, w = starting_b, starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)

    return [b, w]


# data
points = np.concatenate((X, y), axis=1)

# initial values for b and w (model parameters)
b_initial, w_initial = 0, 0
print("starting grandient descent as b: {0}, w: {1}, error: {2}".format(
    b_initial, w_initial, compute_error_for_line_given_points(b_initial, w_initial, points)
))

# run the gradient descent algorithm
b, w = gradient_descent_runner(points, b_initial, w_initial, learning_rate, num_iterations)
print("after {0} iterations, b: {1}, w: {2}, error: {3}".format(
    num_iterations, b, w, compute_error_for_line_given_points(b, w, points)
))

# plot the data and the line
plt.plot(X, y, 'rx')
plt.plot(X, w * X + b, 'b-')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
