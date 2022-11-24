import json
import math
import open3d
import numpy as np


def get_distance(first_point, second_point):
    return math.sqrt(
        (first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2 + (
                first_point[2] - second_point[2]) ** 2)


class NurbsSurface:
    def __init__(self):
        self.control_points = None
        self.data_points = None
        self.n = None
        self.k = None
        self.s_column = None
        self.s_row = None
        self.knots_column = None
        self.knots_row = None

    def basis_function(self, t, i, k, knot_vector):
        if k == 0:
            # if t == knot_vector[i] or knot_vector[i] < t < knot_vector[i + 1]
            if knot_vector[i] <= t <= knot_vector[i + 1]:
                return 1
            return 0
        part_1 = 0.
        denominator_1 = (knot_vector[i + k] - knot_vector[i])
        if denominator_1 != 0:
            part_1 = (t - knot_vector[i]) / denominator_1 * self.basis_function(t, i, k - 1, knot_vector)

        part_2 = 0.
        denominator_2 = (knot_vector[i + k + 1] - knot_vector[i + 1])
        if denominator_2 != 0:
            part_2 = (knot_vector[i + k + 1] - t) / denominator_2 * self.basis_function(t, i + 1, k - 1, knot_vector)

        return part_1 + part_2

    def initialize_data_points(self, points, grid, indices, k):
        self.n = grid[0]
        self.k = k
        self.data_points = np.zeros((self.n, self.n, 3))
        for i, grid_position in enumerate(indices):
            self.data_points[grid_position[0], grid_position[1]] = points[i]

    def find_params_vector(self, points_vector):
        n = self.n
        result = np.zeros(n, dtype=np.float32)
        result[0] = 0.
        result[n - 1] = 1.
        d = sum(get_distance(points_vector[i], points_vector[i - 1]) for i in range(1, n))
        for i in range(1, n - 1):
            result[i] = result[i - 1] + get_distance(points_vector[i], points_vector[i - 1]) / d
        return result

    def find_s_vector(self, parameters_matrix):
        n = self.n
        result = np.zeros(n, dtype=np.float32)
        for i in range(n):
            temp_vector = parameters_matrix[:, i]
            result[i] = sum(temp_vector[j] for j in range(n)) / n
        return result

    def find_s(self):
        n = self.n
        column_matrix = np.zeros((n, n), dtype=np.float32)
        row_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            column_matrix[i] = self.find_params_vector(self.data_points[:, i])
            row_matrix[i] = self.find_params_vector(self.data_points[i])

        self.s_column = self.find_s_vector(column_matrix)
        self.s_row = self.find_s_vector(row_matrix)

    def generate_knots(self):
        k = self.k
        n = self.n
        knots_num = n + k + 1
        self.knots_column = np.zeros(knots_num, dtype=np.float32)
        self.knots_row = np.zeros(knots_num, dtype=np.float32)
        for j in range(1, n - k):
            self.knots_column[j + k] = 1. / k * sum(self.s_column[m] for m in range(j, j + k))
            self.knots_row[j + k] = 1. / k * sum(self.s_row[m] for m in range(j, j + k))
        for j in range(k + 1):
            self.knots_column[j] = 0.
            self.knots_row[j] = 0.
            self.knots_column[knots_num - j - 1] = 1.
            self.knots_row[knots_num - j - 1] = 1.

    def generate_control_points(self):
        n = self.n
        k = self.k
        q_matrix = np.zeros((n, n, 3))
        self.control_points = np.zeros((n, n, 3))
        temp_matrix = np.zeros((n, n), dtype=np.float32)
        # temp_matrix[0, 0] = 1.
        # temp_matrix[n - 1, n - 1] = 1.

        for d in range(n):
            for i in range(n):
                # for j in range(1, n-1):
                for j in range(n):
                    temp_matrix[i, j] = self.basis_function(self.s_column[i], j, k, self.knots_column)
            q_matrix[:, d] = np.linalg.solve(temp_matrix, self.data_points[:, d])

        for c in range(n):
            for i in range(1, n - 1):
                for j in range(n):
                    temp_matrix[i, j] = self.basis_function(self.s_row[i], j, k, self.knots_row)
            self.control_points[c] = np.linalg.solve(temp_matrix, q_matrix[c])

    def get_point_on_surface(self, u, v):
        k = self.k
        n = self.n
        result = np.zeros(3)
        for i in range(n):
            for j in range(n):
                n_i_k = self.basis_function(u, i, k, self.knots_column)
                n_j_k = self.basis_function(v, j, k, self.knots_row)
                result += n_i_k * n_j_k * self.control_points[i, j]

        return result

    def get_surface_mesh(self, points_count):
        surface_points = []
        u = np.linspace(0, 1, points_count)
        v = np.linspace(0, 1, points_count)
        for i in range(len(u)):
            for j in range(len(v)):
                surface_points.append(self.get_point_on_surface(u[i], v[j]))
        return surface_points


def matrix2array(matrix):
    n = len(matrix)
    m = len(matrix[0])
    result_array = np.zeros((n * n, 3))
    for i in range(n):
        for j in range(m):
            result_array[i * n + j] = matrix[i, j]
    return result_array


def get_triangles(n):
    result_array = []
    for i in range(n - 1):
        for j in range(n - 1):
            result_array.append(np.array([i * n + j, i * n + j + 1, i * n + j + n]).astype(np.int32))
            result_array.append(np.array([i * n + j + 1, i * n + j + n, i * n + j + n + 1]).astype(np.int32))

    return np.asarray(result_array)


def get_mesh(surface):
    point_count = 10
    points_array = surface.get_surface_mesh(point_count)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(points_array)
    triangles = get_triangles(point_count)
    mesh.triangles = open3d.utility.Vector3iVector(triangles)
    return mesh


def visualize(surface):
    input_points = open3d.geometry.PointCloud()
    input_points.points = open3d.utility.Vector3dVector(matrix2array(surface.data_points))

    control_points = open3d.geometry.PointCloud()
    control_points.points = open3d.utility.Vector3dVector(matrix2array(surface.control_points))

    open3d.visualization.draw_geometries([input_points, get_mesh(surface)], mesh_show_back_face=True,
                                         mesh_show_wireframe=True)

if __name__ == '__main__':
    filename = "E:/Projects/University/MBCG_Module_2/task_2/resources/1.json"
    file = open(filename)
    data = json.load(file)["surface"]
    points = data["points"]
    indices = data["indices"]
    grid = data["gridSize"]

    surface = NurbsSurface()
    surface.initialize_data_points(points, grid, indices, 3)
    surface.find_s()
    surface.generate_knots()
    surface.generate_control_points()

    visualize(surface)
