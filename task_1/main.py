import json
import cv2
import numpy as np


class BezierCubicSpline():
    def __init__(self):
        self.mPoints = None
        self.mA = None
        self.mB = None

    def fit(self, points):
        self.mPoints = np.array(points, dtype=np.float32)

        n = len(self.mPoints) - 1

        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        P = [2 * (2 * self.mPoints[i] + self.mPoints[i + 1]) for i in range(n)]
        P[0] = self.mPoints[0] + 2 * self.mPoints[1]
        P[n - 1] = 8 * self.mPoints[n - 1] + self.mPoints[n]

        self.mA = np.linalg.solve(C, P)
        self.mB = [0] * n
        for i in range(n - 1):
            self.mB[i] = 2 * self.mPoints[i + 1] - self.mA[i + 1]

        self.mB[n - 1] = (self.mA[n - 1] + self.mPoints[n]) / 2

    def predict(self, p1, a, b, p2, t):
        return pow(1 - t, 3) * p1 + 3 * pow(1 - t, 2) * t * a + 3 * (1 - t) * t * t * b + t * t * t * p2

    def draw(self, img):
        for i in range(0, len(self.mPoints) - 1):
            P1 = self.mPoints[i]
            P2 = self.mPoints[i + 1]
            A = self.mA[i]
            B = self.mB[i]

            t = np.linspace(0, 1, 50)
            for k in range(0, len(t) - 1):
                genP1 = self.predict(P1, A, B, P2, t[k])
                genP2 = self.predict(P1, A, B, P2, t[k + 1])

                cv2.line(img, (int(genP1[0]), int(genP1[1])), (int(genP2[0]), int(genP2[1])), (255, 0, 0), 3)

            cv2.circle(img, (int(P1[0]), int(P1[1])), 7, (0, 255, 0), -1)
            cv2.circle(img, (int(P2[0]), int(P2[1])), 7, (0, 255, 0), -1)


def clearCanvas(img):
    img = 255


def drawPoints(img, points, color, radius):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)


if __name__ == '__main__':
    filename = "E:/Projects/University/MBCG_Module_2/task_1/resources/1.json"
    file = open(filename)
    points = json.load(file)['curve']

    # points.append(points[0])

    cubicSpline = BezierCubicSpline()

    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    drawPoints(img, points, (255, 0, 0), 5)
    cv2.imshow("Points", img)
    cv2.waitKey(0)

    cubicSpline.fit(points)
    cubicSpline.draw(img)
    cv2.imshow("Spline", img)
    cv2.waitKey(0)
