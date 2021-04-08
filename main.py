import utils
import numpy as np
import open3d as o3d
import cv2
import scipy
import sys
sys.path.append("3Dreconstruction")
import structure
from structure import linear_triangulation, compute_P_from_fundamental

from metric_recon_calibrated import bestM2


def bad_rec():
    POINT_CLOUD_DIR = "data_heavy/point_cloud.txt"
    IMG_DIR = "data_light/images/opencv_frame_1.png"
    pairs = utils.read_correspondence_from_dump("data_heavy/corr-exact.txt")
    im = cv2.imread(IMG_DIR)
    points1n = []
    points2n = []
    color = []
    for pair in pairs:
        x, y, x2, y2 = map(float, pair[:4])
        points1n.append([x, y, 1.0])
        points2n.append([x2, y2, 1.0])
        x, y, x2, y2 = map(int, pair[:4])
        color.append(im[x, y])

    mat_F = np.load("data_light/f_mat.npy")
    points1n = np.array(points1n).T
    points2n = np.array(points2n).T

    ep_vec = scipy.linalg.null_space(mat_F.T).squeeze()
    a1, a2, a3 = ep_vec

    skew_mat = np.array([[0, -a3, a2],
                         [a3, 0, -a1],
                         [-a2, a1, 0]])
    skew_mat2 = np.matmul(skew_mat, mat_F)
    full_proj2 = np.hstack([skew_mat2, np.expand_dims(ep_vec, 1)])
    full_proj2 = np.vstack([full_proj2, np.array([0, 0, 0, 1])])
    full_proj1 = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).astype(np.float)

    # full_proj1[:, 2] = full_proj2[:, 2]
    # full_proj1[:, 3] = full_proj2[:, 3]
    # full_proj2[:, 2] = np.array([0, 0, 1, 0])
    # full_proj2[:, 3] = np.array([0, 0, 0, 1])

    tripoints3d = linear_triangulation(points1n, points2n, full_proj1, full_proj2)

    with open(POINT_CLOUD_DIR, "w") as a_file:
        for i in range(tripoints3d.shape[1]):
            x_world, y_world, z_world, _ = tripoints3d[:, i]
            r, g, b = color[i]/255.0
            print(x_world, y_world, z_world, r, g, b, file=a_file)

    pcd = o3d.io.read_point_cloud(POINT_CLOUD_DIR, "xyzrgb")
    o3d.visualization.draw_geometries([pcd])


def metric_rec():
    POINT_CLOUD_DIR = "data_heavy/point_cloud.txt"
    IMG_DIR = "data_light/images/opencv_frame_0.png"
    pairs = utils.read_correspondence_from_dump("data_heavy/corr-exact.txt")
    im = cv2.imread(IMG_DIR)
    points1n = []
    points2n = []
    color = []
    for pair in pairs:
        x, y, x2, y2 = map(float, pair[:4])
        points1n.append([x, y])
        points2n.append([x2, y2])
        x, y, x2, y2 = map(int, pair[:4])
        color.append(im[x, y])

    intrinsic = np.array([[7.9445338402961352e+02, 0., 3.1640138962597587e+02],
                          [0., 7.9383985055902872e+02, 1.8492357505786129e+02],
                          [0., 0., 1.]])
    distor_coeff = np.array([-4.1162469438683935e-01, 3.2255401023497482e-01, 0., 0., 0.])

    h, w = im.shape[:2]
    intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distor_coeff, (w, h), 1, (w, h))

    mat_F = np.load("data_light/f_mat.npy")
    points1n = np.array(points1n)
    points2n = np.array(points2n)

    tripoints3d = bestM2(points1n, points2n, mat_F, intrinsic, intrinsic)[0]

    print("visualizing", tripoints3d.shape)
    with open(POINT_CLOUD_DIR, "w") as a_file:
        for i in range(tripoints3d.shape[0]):
            x_world, y_world, z_world = tripoints3d[i]
            r, g, b = color[i] / 255.0
            print(x_world, y_world, z_world, r, g, b, file=a_file)

    pcd = o3d.io.read_point_cloud(POINT_CLOUD_DIR, "xyzrgb")
    o3d.visualization.draw_geometries([pcd])


def visualize_pc(pc_dir, tripoints3d, color):
    print("visualizing", tripoints3d.shape)
    with open(pc_dir, "w") as a_file:
        for i in range(tripoints3d.shape[1]):
            x_world, y_world, z_world, _ = tripoints3d[:, i]
            r, g, b = color[i] / 255.0
            print(x_world, y_world, z_world, r, g, b, file=a_file)

    pcd = o3d.io.read_point_cloud(pc_dir, "xyzrgb")
    o3d.visualization.draw_geometries([pcd])


def metric_rec2():
    POINT_CLOUD_DIR = "data_heavy/point_cloud.txt"
    IMG_DIR = "data_light/images/opencv_frame_1.png"
    pairs = utils.read_correspondence_from_dump("data_heavy/corr-exact.txt")
    im = cv2.imread(IMG_DIR)
    points1n = []
    points2n = []
    color = []
    for pair in pairs[0:20000]:
        x, y, x2, y2 = map(float, pair[:4])
        points1n.append([x, y, 1.0])
        points2n.append([x2, y2, 1.0])
        x, y, x2, y2 = map(int, pair[:4])
        color.append(im[x2, y2])

    intrinsic = np.array([[7.9445338402961352e+02, 0., 3.1640138962597587e+02],
                          [0., 7.9383985055902872e+02, 1.8492357505786129e+02],
                          [0., 0., 1.]])
    distor_coeff = np.array([ -4.1162469438683935e-01, 3.2255401023497482e-01, 0., 0., 0. ])

    h,  w = im.shape[:2]
    intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distor_coeff, (w, h), 1, (w, h))
    points1n = np.array(points1n).T
    points2n = np.array(points2n).T

    points1n = np.dot(np.linalg.inv(intrinsic), points1n)
    points2n = np.dot(np.linalg.inv(intrinsic), points2n)
    E = structure.compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = None
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i
    assert ind is not None
    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)
    visualize_pc(POINT_CLOUD_DIR, tripoints3d, color)


def metric_rec3():
    POINT_CLOUD_DIR = "data_heavy/point_cloud.txt"
    IMG_DIR = "data_light/images/opencv_frame_0.png"
    pairs = utils.read_correspondence_from_dump("data_heavy/corr-exact.txt")
    im = cv2.imread(IMG_DIR)
    points1n = []
    points2n = []
    color = []
    for pair in pairs:
        x, y, x2, y2 = map(float, pair[:4])
        points1n.append([x, y, 1.0])
        points2n.append([x2, y2, 1.0])
        x, y, x2, y2 = map(int, pair[:4])
        color.append(im[x, y])

    mat_K = np.array([[7.9445338402961352e+02, 0., 3.1640138962597587e+02],
                          [0., 7.9383985055902872e+02, 1.8492357505786129e+02],
                          [0., 0., 1.]])
    distor_coeff = np.array([ -4.1162469438683935e-01, 3.2255401023497482e-01, 0., 0., 0. ])

    h,  w = im.shape[:2]
    mat_K, _ = cv2.getOptimalNewCameraMatrix(mat_K, distor_coeff, (w, h), 1, (w, h))
    mat_K_inv = scipy.linalg.inv(mat_K)
    mat_w = mat_K_inv.T @ mat_K_inv

    mat_F = np.load("data_light/f_mat.npy")
    points1n = np.array(points1n).T
    points2n = np.array(points2n).T

    ep_vec = scipy.linalg.null_space(mat_F.T).squeeze()
    a1, a2, a3 = ep_vec

    skew_mat = np.array([[0, -a3, a2],
                         [a3, 0, -a1],
                         [-a2, a1, 0]])
    skew_mat2 = np.matmul(skew_mat, mat_F)
    full_proj2 = np.hstack([skew_mat2, np.expand_dims(ep_vec, 1)])
    full_proj2 = np.vstack([full_proj2, np.array([0, 0, 0, 1])])
    full_proj1 = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).astype(np.float)

    full_proj1[:, 2] = full_proj2[:, 2]
    full_proj1[:, 3] = full_proj2[:, 3]
    full_proj2[:, 2] = np.array([0, 0, 1, 0])
    full_proj2[:, 3] = np.array([0, 0, 0, 1])

    mat_M = full_proj2[:3, :3]

    mat_A = scipy.linalg.inv(mat_M.T @ mat_w @ mat_M)
    mat_A = np.linalg.cholesky(mat_A)
    mat_A = scipy.linalg.inv(mat_A)
    mat_H = np.hstack([mat_A, np.array([[0], [0], [0]])])
    mat_H = np.vstack([mat_H, np.array([0, 0, 0, 1])])
    # mat_H = scipy.linalg.inv(mat_H)
    tripoints3d = linear_triangulation(points1n, points2n, full_proj1, full_proj2)
    print(tripoints3d.shape)
    with open(POINT_CLOUD_DIR, "w") as a_file:
        for i in range(tripoints3d.shape[1]):
            x_world, y_world, _, z_world = tripoints3d[:, i]
            x_world, y_world, z_world, _ = mat_H @ np.array([x_world, y_world, z_world, 1])
            r, g, b = color[i] / 255.0
            print(x_world, y_world, z_world, r, g, b, file=a_file)

    pcd = o3d.io.read_point_cloud(POINT_CLOUD_DIR, "xyzrgb")
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # bad_rec()
    metric_rec()
