import laspy
import numpy as np
import open3d as o3d
import argparse
from numba import njit
def cmdParser():
    parser = argparse.ArgumentParser(description='Example of usage: \n'
                                                 'python main.py --lasfile=C:\..\..filename.las (str)'
                                                 '--radius=10 (int) --divider=7 (int)')

    parser.add_argument(
        '--lasfile',
        type=str,
        help='path to las file to extract earth points from'
    )
    parser.add_argument(
        '--radius',
        type=int,
        help='type = int \nradius in which earth points are selected (bigger radius - less points, '
             'smaller - more points are selected)'
    )

    parser.add_argument(
        '--divider',
        type=int,
        help='type = int \noptional parameter to filter excess points above the ground'
    )
    my_namespace = parser.parse_args()
    return my_namespace.mtp, my_namespace.src, my_namespace.dst


class Las:
    def __init__(self, path2las: str, radius: float, divider: int, step: int):
        self.las = laspy.read(path2las)
        #point_data = self.las.xyz
        self.points = self.las.xyz #point_data[point_data[:, 2].argsort()].T
        self.header = self.las.header
        self.x_min = self.header.x_min
        self.x_max = self.header.x_max

        self.y_min = self.header.y_min
        self.y_max = self.header.y_max

        self.radius = radius # / self.header.scales.max()
        self.step = step
        self.divider = divider
        self.ground = np.empty(0)
        self.ground_slice = np.empty(0)

    def slice_las(self):
        size = self.points.shape[0]
        steps = [(size * i) // self.divider for i in range(self.divider)]
        planes = np.split(self.points, steps)
        self.ground_slice = planes[1]

    def getFloor(self):
        local_minima = []

        for i in range(self.ground_slice.shape[0] - 1):
            mask = np.sqrt((self.ground_slice[:, 0] - self.ground_slice[i, 0]) ** 2 + (
                    self.ground_slice[:, 1] - self.ground_slice[i, 1]) ** 2) <= self.radius
            if self.ground_slice[i, 2] == np.min(self.ground_slice[mask], axis=0)[2]:
                local_minima.append(tuple(self.ground_slice[i]))
        self.ground = np.array(local_minima)

    def getEarth(self):
        coords = self.points
        first_point = coords[0, :]
        distances = np.sqrt(np.sum((coords - first_point) ** 2, axis=1))


    def getFloor_v2(self, dots):
        local_minima = []
        radius = (self.x_max - self.x_min) * self.radius / self.step
        for i in range(dots.shape[0] - 1):
            mask = np.sqrt((dots[:, 0] - dots[i, 0]) ** 2 + (dots[:, 1] - dots[i, 1]) ** 2) <= radius
            if dots[i, 2] == np.min(dots[mask], axis=0)[2]:
                local_minima.append(tuple(dots[i]))
        self.ground = np.array(local_minima)

    def getEarth_v2(self):
        x0 = self.x_min
        x1 = self.x_max
        y0 = self.y_min
        y1 = self.y_max


        x_steps = np.linspace(x0, x1, self.step, endpoint=False).tolist()
        y_steps = np.linspace(y0, y1, self.step, endpoint=False).tolist()
        for column in x_steps:
            for row in y_steps:
                """
                dots = self.points[np.where(
                    np.logical_and(self.points[:, 0] > column, self.points[:, 0] < x_steps[x_steps.index(column) + 1]))]
                """
                dots = self.select_dots(column, row, x_steps, y_steps)
                self.getFloor_v2(dots)


    def getEarth_v3(self):
        x0 = self.x_min
        x1 = self.x_max
        y0 = self.y_min
        y1 = self.y_max
        self.radius = (x1 - x0) * self.radius / self.step

        x_steps = np.linspace(x0, x1, self.step, endpoint=False).tolist()
        y_steps = np.linspace(y0, y1, self.step, endpoint=False).tolist()
        x_prev = x_steps[0]
        y_prev = y_steps[0]

        for column in x_steps[1::]:
            for row in y_steps[1::]:
                """
                dots = self.points[np.where(
                    np.logical_and(self.points[:, 0] > column, self.points[:, 0] < x_steps[x_steps.index(column) + 1]))]
                """
                dots = self.select_dots(column, x_prev, row, y_prev)
                #print(dots.shape)
                self.getFloor_v2(dots)
                y_prev = y_steps[y_steps.index(row) - 1]
            x_prev = x_steps[x_steps.index(column) - 1]


    def select_dots(self, column, col_prev, row, row_prev):
        condition_X = np.logical_and(self.points[:, 0] <= column, self.points[:, 0] > col_prev)
        condition_Y = np.logical_and(self.points[:, 1] <= row, self.points[:, 1] > row_prev)
        condition_Z = np.full((1, self.points.shape[0]), True, dtype=bool)[0]
        condition_mask = np.logical_and.reduce(np.vstack([condition_X, condition_Y, condition_Z]))
        cond_m_2 = np.reshape(condition_mask, (condition_mask.shape[0], -1))
        dots = []
        for i, e in enumerate(cond_m_2):
            if e == True:
                dots.append(self.points[i])
        # unique, counts = np.unique(condition_mask, return_counts=True)
        # count = dict(zip(unique, counts))
        # dots = self.points[np.where(cond_m_2)]
        return np.array(dots)



    def split(self):
        step_x = (self.header.x_max - self.header.x_min)/self.step
        step_y = (self.header.y_max - self.header.y_min) / self.step
        points = [[] for _ in range(self.step**2)]
        for dot in self.points:
            i = j = 0

            while (i * step_x) + self.header.x_min < dot[0]:
                i += 1
            while (j * step_y) + self.header.y_min < dot[1]:
                j += 1
            i -= 1
            j -= 1
            n = j * self.step + i
            points[n].append(dot)

        return points


    def write_las(self):

        ground_las = laspy.LasData(self.header)
        ground_las.xyz = self.ground.copy()
        ground_las.write("ground_points.las")

    def plot_o3d(self):
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(self.ground)
        o3d.visualization.draw_geometries([geom])


if __name__ == '__main__':
    # laspath, divider, radius = cmdParser()
    laspath = 'C:\\Users\pickles\Downloads\Telegram Desktop\LYSVA_RGB_NIR_9\Lysva_may_PP9_D_G_O.las'
    # laspath = 'C:\\Users\pickles\Downloads\Lysva_12052022_VLS.las'
    las = Las(laspath, radius=1, divider=15, step=5)
    cells = las.split()
    for cell in cells:
        las.getFloor_v2(np.array(cell))
    # las.getEarth_v3()
    # las.write_las()
    las.plot_o3d()
