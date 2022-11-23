import json
import math

import numpy as np


def get_distance(first_point, second_point):
    return math.sqrt(
        (first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2 + (
                first_point[2] - second_point[2]) ** 2)


class NurbsSurface:
    def __init__(self):
        self.knot_vectors = None
        self.basis_functions = None
        self.control_points = None
        self.grid_points = None
        self.distances = None
        self.grid_dimensions = None
        self.basis_degree = None

    # j parameter is to show the desired knot_vector
    def get_basis_function(self, t, i, k, j):
        # TODO is it 0 or 1
        if k == 0:
            if self.knot_vectors[j, i] <= t < self.knot_vectors[j, i + 1] or (
                    self.knot_vectors[j, i] == self.knot_vectors[j, i + 1] and self.knot_vectors[j, i] <= t):
                return 1
            return 0
        part_1 = 0.
        denominator_1 = (self.knot_vectors[j, i + k] - self.knot_vectors[j, i])
        if denominator_1 != 0:
            part_1 = (t - self.knot_vectors[j, i]) / denominator_1 * self.get_basis_function(t, i, k - 1, j)

        part_2 = 0.
        denominator_2 = (self.knot_vectors[j, i + k + 1] - self.knot_vectors[j, i + 1])
        if denominator_2 != 0:
            part_2 = (self.knot_vectors[j, i + k + 1] - t) / denominator_2 * self.get_basis_function(t, i + 1, k - 1, j)

        return part_1 + part_2

    def initialize_points(self, points, grid, indices):
        self.grid_dimensions = grid
        self.grid_points = np.zeros((grid[0], grid[1], 3))
        for i, grid_position in enumerate(indices):
            self.grid_points[grid_position[0], grid_position[1]] = points[i]

    def initialize_distances(self):
        n = self.grid_dimensions[0]
        self.distances = np.zeros((2 * n, n))
        for i in range(n):
            d1 = sum(get_distance(self.grid_points[i, p], self.grid_points[i, p - 1]) for p in
                     range(1, n))
            d2 = sum(get_distance(self.grid_points[p, i], self.grid_points[p - 1, i]) for p in
                     range(1, n))
            self.distances[i, 0] = 0
            self.distances[i + n, 0] = 0
            self.distances[i, n - 1] = 1
            self.distances[i + n, n - 1] = 1
            for p in range(1, n - 1):
                self.distances[i, p] = self.distances[i, p - 1] + get_distance(self.grid_points[i, p],
                                                                               self.grid_points[i, p - 1]) / d1
                self.distances[i + n, p] = self.distances[i + n, p - 1] + get_distance(self.grid_points[p, i],
                                                                                       self.grid_points[p - 1, i]) / d2

    def initialize_knots(self, k):
        self.basis_degree = k
        n = self.grid_dimensions[0]
        knots_num = n + k + 1
        self.knot_vectors = np.zeros((2 * n, knots_num))
        for i in range(n):
            for j in range(1, n - k):
                self.knot_vectors[i, j + k] = 1. / k * sum(self.distances[i, m] for m in range(j, j + k))
                self.knot_vectors[i + n, j + k] = 1. / k * sum(self.distances[i + n, m] for m in range(j, j + k))
            for j in range(k + 1):
                self.knot_vectors[i, j] = 0.
                self.knot_vectors[i + n, j] = 0.
                self.knot_vectors[i, knots_num - j - 1] = 1.
                self.knot_vectors[i + n, n - j - 1] = 1.

    def find_control_points(self):
        n = self.grid_dimensions[0]
        self.control_points = np.zeros((2 * n, n, 3))

        for row in range(n):
            n_matrix1 = np.zeros((n, n), dtype=np.float32)
            n_matrix2 = np.zeros((n, n), dtype=np.float32)
            n_matrix1[0, 0] = 1.
            n_matrix2[0, 0] = 1.
            n_matrix1[n - 1, n - 1] = 1.
            n_matrix2[n - 1, n - 1] = 1.
            for point_index in range(1, n - 1):
                for j in range(n):
                    n_matrix1[point_index, j] = self.get_basis_function(self.distances[row, point_index], j,
                                                                        self.basis_degree, row)
                    n_matrix2[point_index, j] = self.get_basis_function(self.distances[row + n, point_index], j,
                                                                        self.basis_degree, row + n)

            self.control_points[row] = np.linalg.solve(n_matrix1, self.grid_points[row])
            self.control_points[row + n] = np.linalg.solve(n_matrix2, self.grid_points[:, row])


if __name__ == '__main__':
    filename = "D:/Projects/University/MBCG/task_1/resources/1.json"
    file = open(filename)
    data = json.load(file)["surface"]
    points = data["points"]
    indices = data["indices"]
    grid = data["gridSize"]

    surface = NurbsSurface()
    surface.initialize_points(points, grid, indices)
    surface.initialize_distances()
    surface.initialize_knots(3)
    surface.find_control_points()
    print("Stop")
