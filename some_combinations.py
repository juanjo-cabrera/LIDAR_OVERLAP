"""
In this script we assess the labelling of some combinations of a given trajectory
"""


from run_3D_overlap import reader_manager
import numpy as np
import itertools as it
import csv
from config import EXP_PARAMETERS
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import pandas as pd

def process_overlap(keyframe_manager, poses, scan_idx, i):
    pre_process = True

    current_pose = poses[scan_idx].array
    reference_pose = poses[i].array

    if pre_process:
        keyframe_manager.keyframes[scan_idx].pre_process()
        keyframe_manager.keyframes[i].pre_process()

    transformation_matrix = np.linalg.inv(current_pose).dot(reference_pose)
    atb, rmse = keyframe_manager.compute_transformation_local_registration(scan_idx, i, method='point2point',
                                                                           initial_transform=transformation_matrix)
    overlap_pose = keyframe_manager.compute_3d_overlap(scan_idx, i, atb)

    atb, rmse = keyframe_manager.compute_transformation_global_registration(scan_idx, i, method='FPFH')
    overlap_fpfh = keyframe_manager.compute_3d_overlap(scan_idx, i, atb)

    overlap = np.maximum(overlap_pose, overlap_fpfh)
    return overlap, overlap_pose, overlap_fpfh

def downsample(positions, times, deltaxy=5):
    """
    Get odometry times separated by dxy (m) and dth (rad)
    """
    sampled_times = []
    sampled_pos = []
    for ind in range(0, len(positions)):
        pos = positions[ind]
        current_time = times[ind]
        if ind == 0:
            sampled_times.append(ind)
            sampled_pos.append(pos)
            pos_i = pos
        pos_1 = pos

        dxy = np.linalg.norm(pos_1[0:2]-pos_i[0:2])

        if dxy > deltaxy:
            sampled_times.append(current_time)
            sampled_pos.append(pos)
            pos_i = pos_1
    return np.array(sampled_times), np.array(sampled_pos)

def sampled_poses(xys):
    """Visualize the trajectory"""
    # set up plot
    fig, ax = plt.subplots()

    # map poses
    ax.scatter(xys[:, 0], xys[:, 1], c='grey', s=10)
    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Poses')
    plt.show()

def vis_poses(xys):
    """Visualize the trajectory"""
    # set up plot
    fig, ax = plt.subplots()

    # map poses
    ax.scatter(xys[:, 0], xys[:, 1], c='grey', s=10)
    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Poses')
    plt.show()

def get_overlap(reference_time, other_time, reference_timestamps, other_timestamps, overlap):
    i_ref = np.where(reference_timestamps == reference_time)
    i_other = np.where(other_timestamps == other_time)
    match_i = np.intersect1d(i_ref, i_other)

    j_ref = np.where(reference_timestamps == other_time)
    j_other = np.where(other_timestamps == reference_time)
    match_j = np.intersect1d(j_ref, j_other)
    match = np.unique(np.concatenate((match_i, match_j)))
    overlap_selected = overlap[int(match)]
    return overlap_selected




if __name__ == "__main__":
    scan_times, poses, positions, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
    # delta_xy = len(scan_times) / 50
    delta_xy = 50 # metros
    sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)
    vis_poses(sampled_positions)

    df = pd.read_csv(EXP_PARAMETERS.directory + '/all_combinations.csv')
    reference_timestamps = np.array(df["Reference timestamp"])
    other_timestamps = np.array(df["Other timestamp"])
    overlap = np.array(df["Overlap"])

    kd_tree = KDTree(positions)
    # overlaps = []
    for index in range(0, len(sampled_positions)):
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]

        _, indices = kd_tree.query(np.array([sampled_position]), k=50)
        # indices = kd_tree.query_radius(np.array([sampled_position]), r=50)
        # indices = np.array(list(indices))
        nearest_positions = positions[indices]
        nearest_positions = np.squeeze(nearest_positions)
        nearest_times = scan_times[indices].flatten() #para que me salga del tipo (10,)
        overlaps = []
        for nearest_time in nearest_times:
            overlap_s = get_overlap(sampled_time, nearest_time, reference_timestamps, other_timestamps, overlap)
            print(overlap_s)
            overlaps.append(overlap_s)
            hist, bins = np.histogram(np.array(overlaps), bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], density=True)
            counter = hist * len(overlaps) / np.sum(hist)

            print(counter)
        vis_poses(nearest_positions)



        # count, bins, ignored = plt.hist(np.array(overlaps), bins='auto', density=True)
        hist, bins, ignored = plt.hist(np.array(overlaps), bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], density=True)
        counter = hist*len(overlaps)/np.sum(hist)
        print(counter)
        plt.show()






    # scan_indices = np.arange(0, len(scan_times))
    # distances = pdist(pos)
    # distances = sorted(distances)
    # scan_combinations = list(it.combinations(scan_indices, 2))
    # samples = np.random.uniform(np.min(distances), np.max(distances), 5)
    # distances = distances.reshape(1, -1) # if your data has a single feature




    # with open(EXP_PARAMETERS.directory + '/labelling_prueba.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
    #     for idx in scan_indices:
    #         writer.writerow([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
    #     for i in range(0, len(scan_combinations)):
    #         print('Calculated: ', i, 'overlaps out of ', len(scan_combinations))
    #         idx_reference = scan_combinations[i][0]
    #         idx_other = scan_combinations[i][1]
    #         overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
    #         writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh, pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])



