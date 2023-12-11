import laspy
import numpy as np
import open3d as o3d
import argparse


def cmdParser():
    parser = argparse.ArgumentParser(description='Example of usage: \n'
                                                 'python main.py --src=C:\..\..filename.las (str)'
                                                 '--dst=filename.las (str) --display=True (bool)')

    parser.add_argument(
        '--src',
        type=str,
        help='path to las file to extract earth points from'
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='name of las file to store earth points in'
    )
    parser.add_argument(
        '--display',
        type=bool,
        help='shows point cloud if True'
    )
    my_namespace = parser.parse_args()
    return my_namespace.src, my_namespace.dst, my_namespace.display


def read_las(path):
    las = laspy.read(path)
    #print(len(las.points))
    return las.header, las.points[las.classification == 2]


def plot_o3d(earth):
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(earth)
    o3d.visualization.draw_geometries([geom])


def write_las(name, header, points):
    ground_las = laspy.LasData(header)
    ground_las.points = points
    ground_las.write(name)


def main():
    laspath, dest_path, display = cmdParser()
    # dest_path = 'asd.las'
    # display = True
    # laspath = 'Lysva_12052022_VLS.las'
    header, ground = read_las(laspath)
    print(len(ground))
    write_las(dest_path, header, ground)
    if display:
        xyz_ground = laspy.read(dest_path).xyz
        plot_o3d(xyz_ground)


if __name__ == '__main__':
    main()
