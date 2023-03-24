"""
In this script we assess the pairs selection to feed a siamese neural network departing from all overlap combinations
"""


from run_3D_overlap import reader_manager
import numpy as np
import csv
from config import EXP_PARAMETERS
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import random
from scipy import interpolate


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
            # sampled_times.append(ind)
            # sampled_pos.append(pos)
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
    ax.scatter(xys[:, 0], xys[:, 1], c='red', s=10)
    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Poses')
    plt.show()

def plot_overlap(reference_positions, other_positions, overlaps):
    """Visualize the overlap value on trajectory"""
    # set up plot
    fig, ax = plt.subplots()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm)  # cmap="magma"
    mapper.set_array(overlaps)
    colors = np.array([mapper.to_rgba(a) for a in overlaps])

    # sort according to overlap
    indices = np.argsort(overlaps)
    reference_positions = reference_positions[:, 0:2]
    other_positions = other_positions[:, 0:2]

    # pose to evaluate
    # x_actual = xys[scan_idx, 0]
    # y_actual = xys[scan_idx, 1]

    # map poses
    other_positions = other_positions[indices]
    ax.scatter(other_positions[:, 0], other_positions[:, 1], c=colors[indices], s=10)
    # ax.scatter(x_5max, y_5max, c='black', marker='X', s=15)
    ax.scatter(reference_positions[:, 0], reference_positions[:, 1], c='red', marker='X', s=5)
    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Overlap for training')
    cbar = fig.colorbar(mapper, ax=ax)
    cbar.set_label('Overlap', rotation=270, weight='bold')
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
    return overlap_selected, int(match)


def global_uniform_distribution(overlap, size=None):
    inds1 = np.where(overlap <= 0.2)
    inds2 = np.where((overlap > 0.2) & (overlap <= 0.4))
    inds3 = np.where((overlap > 0.4) & (overlap <= 0.6))
    inds4 = np.where((overlap > 0.6) & (overlap <= 0.8))
    inds5 = np.where((overlap > 0.8) & (overlap <= 1))

    len_inds1 = len(inds1[0])
    len_inds2 = len(inds2[0])
    len_inds3 = len(inds3[0])
    len_inds4 = len(inds4[0])
    len_inds5 = len(inds5[0])

    if size is None:
        pairs_to_select = min(len_inds1, len_inds2, len_inds3, len_inds4, len_inds5)
    else:
        pairs_to_select = int(size / 5)

    pairs_selected1 = np.array(random.sample(list(inds1[0]), pairs_to_select))
    pairs_selected2 = np.array(random.sample(list(inds2[0]), pairs_to_select))
    pairs_selected3 = np.array(random.sample(list(inds3[0]), pairs_to_select))
    pairs_selected4 = np.array(random.sample(list(inds4[0]), pairs_to_select))
    pairs_selected5 = np.array(random.sample(list(inds5[0]), pairs_to_select))
    pairs_selected = np.concatenate([pairs_selected1, pairs_selected2, pairs_selected3, pairs_selected4, pairs_selected5])

    return pairs_selected



def partial_uniform_distribution(overlap, combination_selected,  size=None):
    inds1 = np.where(overlap <= 0.2)
    inds2 = np.where((overlap > 0.2) & (overlap <= 0.4))
    inds3 = np.where((overlap > 0.4) & (overlap <= 0.6))
    inds4 = np.where((overlap > 0.6) & (overlap <= 0.8))
    inds5 = np.where((overlap > 0.8) & (overlap <= 1))

    len_inds1 = len(inds1[0])
    len_inds2 = len(inds2[0])
    len_inds3 = len(inds3[0])
    len_inds4 = len(inds4[0])
    len_inds5 = len(inds5[0])

    if size is None:
        pairs_to_select = min(len_inds1, len_inds2, len_inds3, len_inds4, len_inds5)
    else:
        pairs_to_select = int(size / 5)

    pairs_selected1 = np.array(random.sample(list(combination_selected[inds1[0]]), pairs_to_select))
    pairs_selected2 = np.array(random.sample(list(combination_selected[inds2[0]]), pairs_to_select))
    pairs_selected3 = np.array(random.sample(list(combination_selected[inds3[0]]), pairs_to_select))
    pairs_selected4 = np.array(random.sample(list(combination_selected[inds4[0]]), pairs_to_select))
    pairs_selected5 = np.array(random.sample(list(combination_selected[inds5[0]]), pairs_to_select))
    pairs_selected = np.concatenate([pairs_selected1, pairs_selected2, pairs_selected3, pairs_selected4, pairs_selected5])

    return pairs_selected


def interpolate_positions(original_positions, scan_times, N):
    kd_tree = KDTree(positions)
    # Calculamos la distancia total recorrida a lo largo de la trayectoria original
    distances = np.zeros(original_positions.shape[0])
    distances[1:] = np.sqrt(np.sum(np.diff(original_positions, axis=0) ** 2, axis=1))
    total_distance = np.sum(distances)

    # Creamos una función interpolante a partir de la trayectoria original
    interpolator = interpolate.interp1d(distances.cumsum(), original_positions, axis=0)

    # Creamos un array de distancias igualmente espaciadas
    target_distances = np.linspace(0, total_distance, N)

    # Interpolamos las posiciones correspondientes a las distancias seleccionadas
    interpolated_positions = interpolator(target_distances)
    sampled_positions = []
    sampled_times = []
    for sampled_position in interpolated_positions:
        _, indice = kd_tree.query(np.array([sampled_position]), k=1)
        sampled_positions.append(original_positions[indice].reshape(3,))
        sampled_times.append(scan_times[indice])

    return np.array(sampled_positions), np.array(sampled_times).flatten()

def get_uniform_pairs(sampled_positions, sampled_times):
    kd_tree = KDTree(positions)
    pairs_selected = []

    for index in range(0, len(sampled_positions)):
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        _, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
        # indices = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)
        overlaps = []
        combinations_idxs = []
        for nearest_time in nearest_times:
            try:
                overlap_s, combination_selected = get_overlap(sampled_time, nearest_time, reference_timestamps, other_timestamps, overlap)
                overlaps.append(overlap_s)
                combinations_idxs.append(combination_selected)
            except:
                continue

        pairs_selected_i = partial_uniform_distribution(np.array(overlaps), np.array(combinations_idxs))

        pairs_selected.extend(pairs_selected_i)
    pairs_selected = np.unique(np.array(pairs_selected))
    return pairs_selected

def anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap, size=None):

    if size is None:
        print('En none')
        delta_xy = 1  # metros
        sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)
        pairs_selected = get_uniform_pairs(sampled_positions, sampled_times)

    else:
        pairs_selected = []
        i = 185
        while len(pairs_selected) < size:
            sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
            pairs_selected = get_uniform_pairs(sampled_positions, sampled_times)
            i += 1

    # vis_poses(sampled_positions)

    return pairs_selected


def random_distribution(overlap, size=None):
    # generating random samples
    indices = np.arange(0, len(overlap))
    pairs_selected = np.array(random.sample(list(indices), size))
    return pairs_selected


def write_csv(pairs, reference_timestamp, other_timestamp, overlap, reference_x, reference_y, other_x, other_y, name):
    with open(EXP_PARAMETERS.directory + '/' + name + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Reference x", "Reference y", "Other x", "Other y"])
        for idx in pairs:
            writer.writerow([reference_timestamp[idx], other_timestamp[idx], overlap[idx],  reference_x[idx], reference_y[idx], other_x[idx], other_y[idx]])



if __name__ == "__main__":
    scan_times, poses, positions, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)

    df = pd.read_csv(EXP_PARAMETERS.directory + '/all_combinations.csv')
    reference_timestamps = np.array(df["Reference timestamp"])
    other_timestamps = np.array(df["Other timestamp"])
    overlap = np.array(df["Overlap"])
    reference_x = np.array(df["Reference x"])
    other_x = np.array(df["Other x"])
    reference_y = np.array(df["Reference y"])
    other_y= np.array(df["Other y"])

    pairs_selected_globally = global_uniform_distribution(overlap)
    pairs_selected_anchor = anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap,
                                                        size=len(pairs_selected_globally))

    # pairs_selected_anchor = anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap)

    pairs_selected_randomly = random_distribution(overlap, size=len(pairs_selected_anchor))

    write_csv(pairs_selected_globally, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='global_uniform')
    write_csv(pairs_selected_anchor, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='anchor_uniform')
    write_csv(pairs_selected_randomly, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='random')

    print(len(pairs_selected_anchor))




