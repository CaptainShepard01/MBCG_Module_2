import json
import cv2
import numpy as np
import os.path


class BezierCubicSpline:
    def __init__(self):
        self.data_points = None
        self.first_control_points = None
        self.second_control_points = None

    def fit(self, points):
        self.data_points = np.array(points, dtype=np.float32)

        n = len(self.data_points) - 1

        coefficients_matrix = 4 * np.identity(n)
        np.fill_diagonal(coefficients_matrix[1:], 1)
        np.fill_diagonal(coefficients_matrix[:, 1:], 1)
        coefficients_matrix[0, 0] = 2
        coefficients_matrix[n - 1, n - 1] = 7
        coefficients_matrix[n - 1, n - 2] = 2

        points_vector = [2 * (2 * self.data_points[i] + self.data_points[i + 1]) for i in range(n)]
        points_vector[0] = self.data_points[0] + 2 * self.data_points[1]
        points_vector[n - 1] = 8 * self.data_points[n - 1] + self.data_points[n]

        self.first_control_points = np.linalg.solve(coefficients_matrix, points_vector)
        self.second_control_points = [0] * n
        for i in range(n - 1):
            self.second_control_points[i] = 2 * self.data_points[i + 1] - self.first_control_points[i + 1]

        self.second_control_points[n - 1] = (self.first_control_points[n - 1] + self.data_points[n]) / 2

    def get_point_on_curve(self, p1, a, b, p2, t):
        return pow(1 - t, 3) * p1 + 3 * pow(1 - t, 2) * t * a + 3 * (1 - t) * t * t * b + t * t * t * p2

    def draw(self, img):
        for i in range(0, len(self.data_points) - 1):
            start_point = self.data_points[i]
            end_point = self.data_points[i + 1]
            first_control_point = self.first_control_points[i]
            second_control_point = self.second_control_points[i]

            t = np.linspace(0, 1, 50)
            for k in range(0, len(t) - 1):
                interval_start = self.get_point_on_curve(start_point, first_control_point, second_control_point,
                                                         end_point, t[k])
                interval_end = self.get_point_on_curve(start_point, first_control_point, second_control_point,
                                                       end_point, t[k + 1])

                cv2.line(img, (int(interval_start[0]), int(interval_start[1])),
                         (int(interval_end[0]), int(interval_end[1])), (0, 0, 0), 3)

            cv2.circle(img, (int(start_point[0]), int(start_point[1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(end_point[0]), int(end_point[1])), 5, (0, 0, 255), -1)


def draw_points(img, points, color, radius):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)


if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), 'resources', '1.json')

    file = open(filename)
    points = json.load(file)['curve']

    cubicSpline = BezierCubicSpline()

    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    draw_points(img, points, (0, 0, 255), 5)
    cv2.imshow("Points", img)
    cv2.waitKey(0)

    cubicSpline.fit(points)
    cubicSpline.draw(img)
    cv2.imshow("Spline", img)
    cv2.waitKey(0)
