import os

import numpy as np

# set the project path
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, 'data')

# initialize variables
learning_rate = 0.01
num_iterations = 1000
initial_b = 0
initial_w = 0


def generate_data_to_csv(n):
    """ Generate fake data csv, points of the linear equation with noise """
    np.random.seed(0)
    x = np.random.rand(n)
    y = 2 * x + 3 + np.random.randn(n)
    data = np.stack((x, y), axis=1)
    np.savetxt(os.path.join(data_path, "data_fake.csv"), data, delimiter=",")
    return data


def compute_error_for_line_given_points(b, w, points):
    """ loss = (WX + b - Y)^2 """
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    """ w' = w - lr * dloss/dw """

    b_gradient, w_gradient = 0, 0
    n = float(len(points))

    for i in range(len(points)):
        x, y = points[i, 0], points[i, 1]
        b_gradient += -(2 / n) * (y - ((w_current * x) + b_current))  # dloss/db  avg
        w_gradient += -(2 / n) * x * (y - ((w_current * x) + b_current))  # dloss/dw  avg

    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)

    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """ Iterate to optimize """
    b, w = starting_b, starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learning_rate)

    return [b, w]


if __name__ == '__main__':
    generate_data_to_csv(100)  # generate data to csv file
    points = np.loadtxt(os.path.join(data_path, "data_fake.csv"), delimiter=",")  # load data from csv file
    train_data, test_data = points[:80], points[80:]  # split data into training and testing sets

    # train the model
    b, w = gradient_descent_runner(train_data, initial_b, initial_w, learning_rate, num_iterations)
    print("Final b: {}, w: {}".format(b, w))
    # test the model
    test_error = compute_error_for_line_given_points(b, w, test_data)
    print("Error on test data: {}".format(test_error))

    """
    Final b: 3.2940507061348856, w: 1.8881539583398386
    Error on test data: 0.9014630358776691
    """
