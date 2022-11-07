"""
Runs a scanmatching algorithm on the point clouds.

Two variants of the scanmatching are applied.
method A: using the whole points clouds.
method B: segments the ground plane and estimates tz, alpha and beta. Next. tx, ty, gamma are computed using the whole
            point clouds.

Parameters:
    Parameters of the experiment:
        deltaxy, deltath: define the relative movement between scans to be processed.
    Parameters of the scanmatcher:
    class Keyframe():
        self.voxel_size = 0.1 --> the size of the voxels. Pointclouds are sampled_down using this size.
        self.voxel_size_normals = 3*self.voxel_size --> the radius to compute normals on each point.
        self.icp_threshold = 3 --> the distance to associate points in the ICP algorithm.

    TODO:
        Parameters should be stored in a yaml file.
"""
from eurocreader.eurocreader import EurocReader
from graphslam.keyframemanager import KeyFrameManager
from tools.homogeneousmatrix import HomogeneousMatrix
from tools.quaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
from config import PARAMETERS
import matplotlib
import matplotlib.cm as cm
from tools.euler import Euler
import pickle

def vis_gt(scan_idx, xys, overlaps):
    """Visualize the overlap value on trajectory"""
    # set up plot
    fig, ax = plt.subplots()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm)  # cmap="magma"
    mapper.set_array(overlaps)
    colors = np.array([mapper.to_rgba(a) for a in overlaps])

    # sort according to overlap
    indices = np.argsort(overlaps)

    #sort the indices corresponding to the 5 largest values
    n_largest_values = 50
    largest_indices = indices[::-1][:n_largest_values]
    x_5max = xys[largest_indices, 0]
    y_5max = xys[largest_indices, 1]

    # pose to evaluate
    x_actual = xys[scan_idx, 0]
    y_actual = xys[scan_idx, 1]



    # map poses
    xys = xys[indices]
    ax.scatter(xys[:, 0], xys[:, 1], c=colors[indices], s=10)
    ax.scatter(x_5max, y_5max, c='black', marker='X', s=15)
    ax.scatter(x_actual, y_actual, c='red', marker='X', s=5)






    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Demo 4: Generate ground truth for training')
    cbar = fig.colorbar(mapper, ax=ax)
    cbar.set_label('Overlap', rotation=270, weight='bold')
    plt.show()


def vis_poses(scan_i, scan_j, xys):
    """Visualize the trajectory"""
    # set up plot
    fig, ax = plt.subplots()

    # poses to evaluate
    x_i = xys[scan_i, 0]
    y_i = xys[scan_i, 1]

    x_j = xys[scan_j, 0]
    y_j = xys[scan_j, 1]


    # map poses
    ax.scatter(xys[:, 0], xys[:, 1], c='grey', s=10)
    ax.scatter(x_i, y_i, c='red', marker='X', s=15)
    ax.scatter(x_j, y_j, c='blue', marker='X', s=15)


    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Poses')
    plt.show()


def compute_homogeneous_transforms(gt_pos, gt_orient):
    transforms = []
    for i in range(len(gt_pos)):
        # CAUTION: THE ORDER IN THE QUATERNION class IS [qw, qx qy qz]
        # the order in ROS is [qx qy qz qw]
        q = [gt_orient[i][3], gt_orient[i][0], gt_orient[i][1], gt_orient[i][2]]
        Q = Quaternion(q)
        Ti = HomogeneousMatrix(gt_pos[i], Q)
        transforms.append(Ti)
    return transforms


def compute_homogeneous_transforms_relative(transforms):
    transforms_relative = []
    # compute relative transformations
    for i in range(len(transforms) - 1):
        Ti = transforms[i]
        Tj = transforms[i + 1]
        Tij = Ti.inv() * Tj
        transforms_relative.append(Tij)
    return transforms_relative


def eval_errors(ground_truth_transforms, measured_transforms):
    # compute xyz alpha beta gamma
    gt_tijs = []
    meas_tijs = []
    for i in range(len(ground_truth_transforms)):
        gt_tijs.append(ground_truth_transforms[i].t2v(n=3))  # !!! convert to x y z alpha beta gamma
        meas_tijs.append(measured_transforms[i].t2v(n=3))

    gt_tijs = np.array(gt_tijs)
    meas_tijs = np.array(meas_tijs)
    errors = gt_tijs-meas_tijs

    plt.figure()
    plt.plot(range(len(errors)), errors[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.title('Errors XYZ')
    plt.show(block=True)

    plt.figure()
    plt.plot(range(len(errors)), errors[:, 3], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 4], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 5], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.title('Errors Alfa Beta Gamma')
    plt.show(block=True)

    print("Covariance matrix: ")
    print(np.cov(errors.T))


def view_pos_data(data):
    plt.figure()
    plt.plot(range(len(data)), data[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(data)), data[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(data)), data[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.show(block=True)

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.show(block=True)


def view_orient_data(data):
    eul = []
    for dat in data:
        q = [dat[3], dat[0], dat[1], dat[2]]
        Q = Quaternion(q)
        th = Q.Euler()
        eul.append(th.abg)
    eul = np.array(eul)

    plt.figure()
    plt.plot(range(len(eul)), eul[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(eul)), eul[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(eul)), eul[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    # plt.legend()
    plt.show(block=True)


def save_transforms_to_file(transforms):
    import pickle
    pickle.dump(transforms, open('measured_transforms.pkl', "wb"))


def spherical_projection(homogeneous_points, fov=45, proj_W=512, proj_H=128, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
        Returns:
           proj_range: projected range image with depth, each pixel contains the corresponding depth
           proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
           proj_intensity: each pixel contains the corresponding intensity
           proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
     """
    fov = np.pi * fov/180.0 #pasamos a radianes
    depth = np.linalg.norm(homogeneous_points[:, :3], 2, axis=1)
    homogeneous_points = homogeneous_points[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = homogeneous_points[:, 0]
    scan_y = homogeneous_points[:, 1]
    scan_z = homogeneous_points[:, 2]
    intensity = homogeneous_points[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    # proj_y = 0.5 * (pitch / (fov/2) + 1.0)  # in [0.0, 1.0] MI INTERPRETACIÓN
    proj_y = 1.0 - (pitch + abs(fov/2)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # tambien np.round
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx

def main():
    directory = PARAMETERS.directory
    # Prepare data
    euroc_read = EurocReader(directory=directory)
    # nmax_scans to limit the number of scans in the experiment
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=PARAMETERS.exp_deltaxy, deltath=PARAMETERS.exp_deltath, nmax_scans=PARAMETERS.exp_long)
    start = 0
    end = 1000
    scan_times = scan_times[start:end]
    gt_pos = gt_pos[start:end]
    gt_orient = gt_orient[start:end]
    # view_pos_data(gt_pos)
    gt_poses = compute_homogeneous_transforms(gt_pos, gt_orient)
    # gt_relative_poses = compute_homogeneous_transforms_relative(gt_poses)

    overlaps = []
    overlaps1 = []
    scan_idx = 100
    pre_process = True

    # create KeyFrameManager
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_all_keyframes()
    keyframe_manager.load_pointclouds()
    if pre_process == True:
        keyframe_manager.keyframes[scan_idx].pre_process()
    current_homogeneous_points = keyframe_manager.keyframes[scan_idx].points2homogeneous(pre_process=False)
    current_range, project_points, _, _ = spherical_projection(current_homogeneous_points)
    # plt.imshow(current_range)
    # plt.imshow(project_points)
    # plt.show()

    detected_points = project_points[current_range > 0]
    # valid_num = len(visible_points)
    # current_pose = gt_poses[scan_idx].array
    current_pose = np.eye(4)
    keyframe_manager.keyframes[scan_idx].pre_process()
    # keyframe_manager.keyframes[scan_idx].draw_pointcloud()
    # keyframe_manager.keyframes[0].pre_process()

    saved_overlap = False
    debug = False

    if saved_overlap == True:
        with open("overlaps_pi_sextos", "rb") as fp:  # Unpickling
            overlaps = pickle.load(fp)
        xys = gt_pos[:, 0:2]



    else:

        for i in range(0, len(scan_times)):
            print('Adding keyframe and computing transform: ', i, 'out of ', len(scan_times))
            overlaps_i = []
            gammas = np.arange(0, 2 * np.pi, np.pi/6)

            if debug:
                # i = 150
                xys = gt_pos[:, 0:2]
                vis_poses(scan_idx, i, xys)

            for gamma in gammas:
                tx = ty = tz = beta = alpha = 0
                transformation_matrix = HomogeneousMatrix(np.array([tx, ty, tz]), Euler(np.array([alpha, beta, gamma])))
                transformation_matrix = transformation_matrix.array
                # print(transformation_matrix)

                keyframe_manager.keyframes[i].pre_process()

                if debug:
                    keyframe_manager.keyframes[i].draw_registration_result(keyframe_manager.keyframes[scan_idx], transformation_matrix)

                atb, rmse = keyframe_manager.compute_transformation_local_registration(scan_idx, i, method='B', initial_transform=transformation_matrix)
                # measured_transforms.append(atb)

                # reference_pose = gt_poses[i].array
                if pre_process == True:
                    keyframe_manager.keyframes[i].pre_process()
                reference_homogeneous_points = keyframe_manager.keyframes[i].points2homogeneous(pre_process=False)

                # reference_points_world = reference_pose.dot(reference_homogeneous_points.T).T
                # reference_points_in_current = np.linalg.inv(current_pose).dot(reference_points_world.T).T
                # reference_points_in_current = HomogeneousMatrix.__mul__(atb, reference_homogeneous_points.T).T
                reference_points_in_current = np.dot(atb.array, reference_homogeneous_points.T).T
                reference_range, reference_project_points, _, _ = spherical_projection(reference_points_in_current)
                reference_detected_points = reference_project_points[reference_range > 0]  # filtra los puntos que dan en el infinito y devuelven -1
                valid_num = np.minimum(len(detected_points), len(reference_detected_points))

                # fig = plt.figure()
                # rows = 2
                # columns = 1
                # fig.add_subplot(rows, columns, 1)
                # # showing image
                # plt.imshow(current_range, cmap='gray')
                # plt.axis('off')
                # plt.title("Current")
                # fig.add_subplot(rows, columns, 2)
                # # showing image
                # plt.imshow(reference_range, cmap='gray')
                # plt.axis('off')
                # plt.title("Reference")
                #
                # plt.show()


                overlap_i = np.count_nonzero(
                    abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < 1) / valid_num
                overlaps_i.append(overlap_i)
                # print(overlap_i)
                # if overlap_i == 1:
                #     break


            overlap = max(overlaps_i)
            overlaps.append(overlap)
            # print(overlap)
            # xys = gt_pos[:, 0:2]
            # vis_gt(xys, overlaps)

        xys = gt_pos[:, 0:2]
        with open("metodo2_overlaps_pi_sextos", "wb") as fp:
            pickle.dump(overlaps, fp)

    vis_gt(scan_idx, xys, overlaps)

if __name__ == "__main__":
    main()
