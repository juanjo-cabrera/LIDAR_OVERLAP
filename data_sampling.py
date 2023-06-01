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
from scipy.interpolate import lagrange

df = pd.read_csv(EXP_PARAMETERS.directory + '/all_combinations.csv')
reference_timestamps = np.array(df["Reference timestamp"])
other_timestamps = np.array(df["Other timestamp"])
overlap = np.array(df["Overlap"])
reference_x = np.array(df["Reference x"])
other_x = np.array(df["Other x"])
reference_y = np.array(df["Reference y"])
other_y = np.array(df["Other y"])


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
    """
     Uniform distribution with non-repetitive combinations, it is not necessary to check it
     """
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
    """
    Uniform distribution per anchor without checking repetitive combinations, it is done in a posterior line which
    modify the uniform distribution to partial distribution per anchor
    """
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

    pairs_selected = np.concatenate([pairs_selected5, pairs_selected4, pairs_selected3, pairs_selected2, pairs_selected1])

    return pairs_selected

def partial_uniform_distribution2(overlap, combination_selected, distances, size=None):
    """
    Uniform distribution per anchor without checking repetitive combinations, it is done in a posterior line which
    modify the uniform distribution to partial distribution per anchor
    """
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
    elif size == 'max':
        pairs_to_select = None #solucion temporal
    else:
        pairs_to_select = int(size)

    try:
        if size == 'max':
            pairs_to_select = len_inds1
        inds_selected1 = np.array(random.sample(list(inds1[0]), pairs_to_select))
        pairs_selected1 = combination_selected[inds_selected1]
        distances1 = distances[inds_selected1]
        # pairs_selected1 = np.array(random.sample(list(combination_selected[inds1[0]]), pairs_to_select))
    except:
        pairs_selected1 = None
    try:
        if size == 'max':
            pairs_to_select = len_inds2
        inds_selected2 = np.array(random.sample(list(inds2[0]), pairs_to_select))
        pairs_selected2 = combination_selected[inds_selected2]
        distances2 = distances[inds_selected2]
        # pairs_selected2 = np.array(random.sample(list(combination_selected[inds2[0]]), pairs_to_select))
    except:
        pairs_selected2 = None
    try:
        if size == 'max':
            pairs_to_select = len_inds3
        inds_selected3 = np.array(random.sample(list(inds3[0]), pairs_to_select))
        pairs_selected3 = combination_selected[inds_selected3]
        distances3 = distances[inds_selected3]
        # pairs_selected3 = np.array(random.sample(list(combination_selected[inds3[0]]), pairs_to_select))
    except:
        pairs_selected3 = None
    try:
        if size == 'max':
            pairs_to_select = len_inds4
        inds_selected4 = np.array(random.sample(list(inds4[0]), pairs_to_select))
        pairs_selected4 = combination_selected[inds_selected4]
        distances4 = distances[inds_selected4]
        # pairs_selected4 = np.array(random.sample(list(combination_selected[inds4[0]]), pairs_to_select))
    except:
        pairs_selected4 = None
    try:
        if size == 'max':
            pairs_to_select = len_inds5
        inds_selected5 = np.array(sorted(random.sample(list(inds5[0]), pairs_to_select)))
        pairs_selected5 = combination_selected[inds_selected5]
        distances5 = distances[inds_selected5]
        # pairs_selected5 = combination_selected[np.array(sorted(random.sample(list(inds5[0]), pairs_to_select)))]
    except:
        pairs_selected5 = None

    return pairs_selected5, pairs_selected4, pairs_selected3, pairs_selected2, pairs_selected1, distances5, distances4, distances3, distances2, distances1

def total_uniform_distribution(overlap, combination_proposed, already_selected, size=None):
    """
     Uniform distribution per anchor checking repetitive combitnations, it is done in a posterior line which
     modify the uniform distribution to partial distribution per anchor
     """
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
    len_inds = np.array([len_inds1, len_inds2, len_inds3, len_inds4, len_inds5])

    if size is None:
        N_pairs_to_select = min(len_inds1, len_inds2, len_inds3, len_inds4, len_inds5)
    else:
        N_pairs_to_select = int(size / 5)

    while True:
        pairs_selected1 = np.array(random.sample(list(combination_proposed[inds1[0]]), N_pairs_to_select))
        pairs_selected2 = np.array(random.sample(list(combination_proposed[inds2[0]]), N_pairs_to_select))
        pairs_selected3 = np.array(random.sample(list(combination_proposed[inds3[0]]), N_pairs_to_select))
        pairs_selected4 = np.array(random.sample(list(combination_proposed[inds4[0]]), N_pairs_to_select))
        pairs_selected5 = np.array(random.sample(list(combination_proposed[inds5[0]]), N_pairs_to_select))

        # flag1 = np.where(already_selected == pairs_selected1)[0]
        # flag2 = np.where(already_selected == pairs_selected2)[0]
        # flag3 = np.where(already_selected == pairs_selected3)[0]
        # flag4 = np.where(already_selected == pairs_selected4)[0]
        # flag5 = np.where(already_selected == pairs_selected5)[0]

        flag1 = np.intersect1d(already_selected, pairs_selected1)
        flag2 = np.intersect1d(already_selected, pairs_selected2)
        flag3 = np.intersect1d(already_selected, pairs_selected3)
        flag4 = np.intersect1d(already_selected, pairs_selected4)
        flag5 = np.intersect1d(already_selected, pairs_selected5)


        flags = np.array([flag1, flag2, flag3, flag4, flag5])

        pairs_to_select = np.concatenate([pairs_selected1, pairs_selected2, pairs_selected3, pairs_selected4, pairs_selected5])
        acumulated_pairs = np.concatenate([already_selected, pairs_to_select])

        if len(np.unique(acumulated_pairs)) == len(acumulated_pairs):
            break
        # if len(flag1) == 0 and len(flag2) == 0 and len(flag3) == 0 and len(flag4) == 0 and len(flag5) == 0:
        #     break
        else:
            for i in range(0, len(flags)):
                flag = flags[i]
                len_ind = len_inds[i]
                if len(flag) != 0:
                    # if len_ind > N_pairs_to_select:
                        # break
                    # if len_ind == N_pairs_to_select:
                        # N_pairs_to_select = N_pairs_to_select - 1
                        # len_inds[i] = len_inds[i] - 1
                        indexes_to_conserve = np.where(flag != combination_selected)
                        combination_selected = combination_selected[indexes_to_conserve[0]]
                        overlap = overlap[indexes_to_conserve[0]]
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
                        len_inds = np.array([len_inds1, len_inds2, len_inds3, len_inds4, len_inds5])
                        N_pairs_to_select = min(len_inds1, len_inds2, len_inds3, len_inds4, len_inds5)





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
    # Buscamos los puntos de la trayectoria real que más cerca están de los interpolados
    for sampled_position in interpolated_positions:
        _, indice = kd_tree.query(np.array([sampled_position]), k=1)
        sampled_positions.append(original_positions[indice].reshape(3,))
        sampled_times.append(scan_times[indice])

    return np.array(sampled_positions), np.array(sampled_times).flatten()

def get_partial_uniform_pairs(sampled_positions, sampled_times):
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
    pairs_selected = np.unique(np.array(pairs_selected))  # Esta linea de aqui es la que hace que la distribucion no quede 100% uniforme
    return pairs_selected


def get_total_uniform_pairs(sampled_positions, sampled_times):
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
        combinations_proposed = []
        for nearest_time in nearest_times:
            try:
                overlap_s, combination_proposed = get_overlap(sampled_time, nearest_time, reference_timestamps, other_timestamps, overlap)
                overlaps.append(overlap_s)
                combinations_proposed.append(combination_proposed)
            except:
                continue


        if index == 0:
            pairs_selected_i = partial_uniform_distribution(np.array(overlaps), np.array(combinations_proposed))
            pairs_selected.extend(pairs_selected_i)
        else:
            common_combinations = np.intersect1d(pairs_selected, combinations_proposed)
            if len(common_combinations) > 0:
                import warnings
                warnings.filterwarnings("ignore", message="elementwise comparison failed; this will raise an error in the future.")
                indexes_to_conserve = np.where(common_combinations != combinations_proposed)
                combinations_proposed = np.array(combinations_proposed)[indexes_to_conserve[0].astype(int)]
                overlaps = np.array(overlaps)[indexes_to_conserve[0].astype(int)]
            pairs_selected_i = partial_uniform_distribution(np.array(overlaps), np.array(combinations_proposed))
            pairs_selected.extend(pairs_selected_i)

    return pairs_selected




def anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap, size=None):

    if size is None:
        print('En none')
        delta_xy = 1  # metros
        sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)
        pairs_selected = get_total_uniform_pairs(sampled_positions, sampled_times)

    else:
        pairs_selected = []
        i = 150
        while len(pairs_selected) < size:
            sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
            # pairs_selected = get_partial_uniform_pairs(sampled_positions, sampled_times)
            pairs_selected = get_total_uniform_pairs(sampled_positions, sampled_times)
            print(len(pairs_selected))
            i += 1


    # vis_poses(sampled_positions)

    return pairs_selected

def anchor_partial_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap, size):


    i = 186
    pairs_selected = []
    while len(pairs_selected) < size:
        sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
        pairs_selected = get_partial_uniform_pairs(sampled_positions, sampled_times)
        # pairs_selected = get_total_uniform_pairs(sampled_positions, sampled_times)
        print(len(pairs_selected))
        i += 1

    return pairs_selected


def fill_predictor(sampled_positions, sampled_times, kd_tree, sample_storage, distance_overlap, csv_overlap):

    sampled_position = sampled_positions[0]
    sampled_time = sampled_times[0]
    distances, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
    distances = distances.flatten()
    # indices, distances = kd_tree.query_radius(np.array([sampled_position]), r=5)
    indices = np.array(list(indices))
    nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)

    for i in range(0, len(nearest_times), 5):
        # try:
        nearest_time = nearest_times[i]
        distance = distances[i]
        overlap, combination_proposed = get_overlap(sampled_time, nearest_time, reference_timestamps, other_timestamps, csv_overlap)
        distance_overlap.add(overlap, distance)
        sample_storage.add(sampled_time, nearest_time, overlap)
        # except:
        #     continue

def fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances):

    for i in range(0, len(csv_distances)):
        distance = csv_distances[i]
        overlap = csv_overlap[i]
        distance_overlap.add(overlap, distance)




def get_online_pairs(sampled_positions, sampled_times, csv_overlap):
    sample_storage = SampleStorage()
    distance_overlap = DistanceOverlap_Relation()
    kd_tree = KDTree(positions)
    pairs_selected = []

    fill_predictor(sampled_positions, sampled_times, kd_tree, sample_storage, distance_overlap, csv_overlap)
    n_to_fill = distance_overlap.get_len()
    print('Nº ejemplos previos: ', n_to_fill)
    suma = 0
    for index in range(0, len(sampled_positions)):
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        distances, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
        distances = distances.flatten()
        # indices, distances = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)
        overlaps = []
        combinations_proposed = []
        sample_admin = SampleAdministrator()

        overlap_predicted = distance_overlap.predict_overlap(distances)

        distance_overlap.plot_tendency()

        times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted), nearest_times, distances, size='max')

        # nearest_times_selected = nearest_times[i_pairs]

        # for i in range(0, len(nearest_times_selected)):
        skip_to = None
        flag8 = flag6 = flag4 = flag2 = flag0 = False
        while skip_to != -1:
            try:
                skip_to = None
                len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
                max_len = np.max([len0, len2, len4, len6, len8])
                if len8 == 0 and flag8 == False:
                    i = 0
                    flag8 = True
                    # while skip_to == None:
                    while skip_to != 6 and skip_to != 4:
                        nearest_time = times8[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                      reference_timestamps, other_timestamps,
                                                                      overlap)
                        distance_overlap.add(overlap_candidate, distances8[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1
                elif len6 < max_len and flag6 == False:
                    # len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
                    # when index = 2, bug here because all the candidates belong to 0.8 - 1.
                    # infinite loop
                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, len8)
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances)
                    i = 0
                    flag6 = True
                    # while skip_to == None:
                    while skip_to != 4:
                        nearest_time = times6[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        distance_overlap.add(overlap_candidate, distances6[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1

                elif len4 < max_len and flag4 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag4 = True
                    # while skip_to == None:
                    while skip_to != 2:
                        nearest_time = times4[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        distance_overlap.add(overlap_candidate, distances4[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1

                elif len2 < max_len and flag2 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag2 = True
                    # while skip_to == None:
                    while skip_to != 0:
                        nearest_time = times2[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        distance_overlap.add(overlap_candidate, distances2[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1



                elif len0 < max_len and flag0 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag0 = True
                    # while skip_to == None:
                    while skip_to != -1:
                        nearest_time = times0[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        distance_overlap.add(overlap_candidate, distances0[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1
                    break
                else:
                    skip_to = -1


                # overlaps.append(overlap_s)
                # combinations_proposed.append(combination_proposed)
            except:
                continue
        combinations_selected_i = sample_admin.get_combinations()
        pairs_selected.extend(combinations_selected_i)
        print('Combinations_selected: ', pairs_selected)
        len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
        print('Index: ', index, 'lens: ', [len0, len2, len4, len6, len8])
        suma = suma + np.sum(np.array([len0, len2, len4, len6, len8]))
        print('Nº ejemplos selecciondados: ', suma, 'Nº ejemplos calculados: ', distance_overlap.get_len() - n_to_fill, 'Nº ejemplos desaprovechados: ', distance_overlap.get_len() - suma - n_to_fill)

    pairs_selected = np.unique(np.array(pairs_selected))  # Esta linea de aqui es la que hace que la distribucion no quede 100% uniforme
    return pairs_selected


def get_online_pairs_ALL_INFO(sampled_positions, sampled_times, csv_overlap, csv_distances):
    sample_storage = SampleStorage()
    distance_overlap = DistanceOverlap_Relation()
    kd_tree = KDTree(positions)
    pairs_selected = []

    fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances)
    n_to_fill = distance_overlap.get_len()
    print('Nº ejemplos previos: ', n_to_fill)
    suma = 0
    for index in range(0, len(sampled_positions)):
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        distances, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
        distances = distances.flatten()
        # indices, distances = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)
        overlaps = []
        combinations_proposed = []
        sample_admin = SampleAdministrator()

        overlap_predicted = distance_overlap.predict_overlap(distances)

        distance_overlap.plot_tendency()
        distance_overlap.plot_occupancy_grid()

        times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted), nearest_times, distances, size='max')

        # nearest_times_selected = nearest_times[i_pairs]

        # for i in range(0, len(nearest_times_selected)):
        skip_to = None
        flag8 = flag6 = flag4 = flag2 = flag0 = False
        while skip_to != -1:
            try:
                skip_to = None
                len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
                max_len = np.max([len0, len2, len4, len6, len8])
                if len8 == 0 and flag8 == False:
                    i = 0
                    flag8 = True
                    # while skip_to == None:
                    while skip_to != 6 and skip_to != 4:
                        nearest_time = times8[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                      reference_timestamps, other_timestamps,
                                                                      overlap)
                        # distance_overlap.add(overlap_candidate, distances8[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1
                elif len6 < max_len and flag6 == False:
                    # len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
                    # when index = 2, bug here because all the candidates belong to 0.8 - 1.
                    # infinite loop
                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, len8)
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances)
                    i = 0
                    flag6 = True
                    # while skip_to == None:
                    while skip_to != 4:
                        nearest_time = times6[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        # distance_overlap.add(overlap_candidate, distances6[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1

                elif len4 < max_len and flag4 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag4 = True
                    # while skip_to == None:
                    while skip_to != 2:
                        nearest_time = times4[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        # distance_overlap.add(overlap_candidate, distances4[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1

                elif len2 < max_len and flag2 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag2 = True
                    # while skip_to == None:
                    while skip_to != 0:
                        nearest_time = times2[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        # distance_overlap.add(overlap_candidate, distances2[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1



                elif len0 < max_len and flag0 == False:

                    # overlap_predicted = distance_overlap.predict_overlap(distances)
                    # distance_overlap.plot_tendency()
                    # times8, times6, times4, times2, times0, distances8, distances6, distances4, distances2, distances0 = partial_uniform_distribution2(np.array(overlap_predicted),
                    #                                                                        nearest_times, distances, len8)
                    i = 0
                    flag0 = True
                    # while skip_to == None:
                    while skip_to != -1:
                        nearest_time = times0[i]
                        overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_time,
                                                                              reference_timestamps, other_timestamps,
                                                                              overlap)
                        # distance_overlap.add(overlap_candidate, distances0[i])
                        skip_to = sample_admin.manage_overlap(overlap_candidate, combination_proposed)
                        i += 1
                    break
                else:
                    skip_to = -1


                # overlaps.append(overlap_s)
                # combinations_proposed.append(combination_proposed)
            except:
                continue
        combinations_selected_i = sample_admin.get_combinations()
        pairs_selected.extend(combinations_selected_i)
        print('Combinations_selected: ', pairs_selected)
        len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
        print('Index: ', index, 'lens: ', [len0, len2, len4, len6, len8])
        suma = suma + np.sum(np.array([len0, len2, len4, len6, len8]))
        print('Nº ejemplos selecciondados: ', suma, 'Nº ejemplos calculados: ', distance_overlap.get_len() - n_to_fill, 'Nº ejemplos desaprovechados: ', distance_overlap.get_len() - suma - n_to_fill)

    pairs_selected = np.unique(np.array(pairs_selected))  # Esta linea de aqui es la que hace que la distribucion no quede 100% uniforme
    return pairs_selected


def plot_hist(bin_edges, pvalues, width):
    bin_centers = np.zeros_like(pvalues)
    for i in range(len(bin_edges)-1):
        bin_centers[i] = (bin_edges[i]+bin_edges[i+1])/2
    plt.bar(x=bin_centers, height=pvalues, width=width)
    plt.show()

def get_histogram(actual_distribution, N):
    pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
    counter = pvalues * len(actual_distribution) / np.sum(pvalues)
    return counter

def get_online_grid_ALL_INFO(sampled_positions, sampled_times, csv_overlap, csv_distances):
    sample_storage = SampleStorage()
    distance_overlap = DistanceOverlap_Relation()
    kd_tree = KDTree(positions)
    pairs_selected = []

    fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances)
    n_to_fill = distance_overlap.get_len()
    print('Nº ejemplos previos: ', n_to_fill)
    suma = 0

    for index in range(0, len(sampled_positions)):

        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        distances, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
        distances = distances.flatten()
        # indices, distances = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)
        actual_distribution = []
        combinations_proposed = []
        sample_admin = SampleAdministrator()

        # overlap_predicted = distance_overlap.predict_overlap(distances)

        """
        distance_overlap.plot_tendency()
        distance_overlap.plot_occupancy_grid()
   
        """

        N = 5 # number of bins
        N_max = 50 # number of examples per anchor

        # distribution_goal = np.random.uniform(0.9, 1, size=N_max)
        # distribution_goal = list(distribution_goal)
        # pvalues, bin_edges = np.histogram(distribution_goal, bins=np.linspace(0, 1, N + 1), density=True)
        # sa = np.std(pvalues, axis=0)
        # print("Uniformidad inicial: ", sa)
        # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        #
        #
        # while len(distribution_goal) > 0:
        #     overlap_goal = distribution_goal.pop(0)
        #     min_distance, max_distance = distance_overlap.distances2search(overlap_goal)
        #     pairs_candidate = np.where(np.bitwise_and(distances < max_distance, distances >= min_distance))[0]
        #     try:
        #         pair_candidate = np.random.choice(pairs_candidate)
        #     except:
        #         break
        #
        #     overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_times[pair_candidate],
        #                                                           reference_timestamps, other_timestamps,
        #                                                           overlap)
        #     sample_admin.save_overlap(overlap_candidate)
        #     sample_admin.save_candidate(combination_proposed)
        #
        #     print(overlap_candidate)
        #     actual_distribution.append(overlap_candidate)
        #     distances = np.delete(distances, pair_candidate)
        #     nearest_times = np.delete(nearest_times, pair_candidate)
        #
        # pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
        # si = np.std(pvalues, axis=0)
        # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        # index = np.argmin(pvalues)
        # goal_updated = float(np.random.uniform(bin_edges[index], bin_edges[index + 1], 1))

        sa = 0.1
        si = 1000
        # pvalues, bin_edges = np.histogram(np.zeros(N), bins=np.linspace(0, 1, N + 1), density=True)
        # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        #Vecino mas cercano
        pair_candidate = 0
        pairs = np.where(distances <= 2)[0]
        for pair_candidate in pairs:
            overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_times[pair_candidate],
                                                                  reference_timestamps, other_timestamps,
                                                                  overlap)
            sample_admin.save_overlap(overlap_candidate)
            sample_admin.save_candidate(combination_proposed)

            print(overlap_candidate)
            actual_distribution.append(overlap_candidate)
            distances = np.delete(distances, pair_candidate)
            nearest_times = np.delete(nearest_times, pair_candidate)

        uniformidad = []
        pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
        pvalues = pvalues * len(actual_distribution) / np.sum(pvalues)
        """
        # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        """
        min_value_initial = pvalues[N-1].min()
        min_value = pvalues.min()
        si = np.std(pvalues, axis=0)
        uniformidad.append(si)


        tendency = -1
        ayuda = 0
        while tendency < 0 or round(min_value) != round(min_value_initial):
            if round(min_value) == round(min_value_initial):
                break

            indexes = np.where(pvalues == pvalues.min())[0]
            index = np.max(indexes)
            # new samples, generate arbitrarily 1 or more at each iteration

            goal_updated = float(np.random.uniform(bin_edges[index], bin_edges[index + 1], 1))

            print('expectation: ', goal_updated)
            # distribution_goal = list(np.append(actual_distribution, goals_updated))



            min_distance, max_distance = distance_overlap.distances2search(goal_updated)
            pairs_candidate = np.where(np.bitwise_and(distances < max_distance, distances > min_distance))[0]
            try:
                pair_candidate = np.random.choice(pairs_candidate)

                overlap_candidate, combination_proposed = get_overlap(sampled_time, nearest_times[pair_candidate],
                                                                      reference_timestamps, other_timestamps,
                                                                      overlap)
                print('reality: ', overlap_candidate)
                sample_admin.save_overlap(overlap_candidate)
                sample_admin.save_candidate(combination_proposed)

                print(overlap_candidate)
                actual_distribution.append(overlap_candidate)
                distances = np.delete(distances, pair_candidate)
                nearest_times = np.delete(nearest_times, pair_candidate)

                pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
                pvalues = pvalues * len(actual_distribution) / np.sum(pvalues)
                min_value = pvalues.min()

                si = np.std(pvalues, axis=0)
                uniformidad.append(si)
                tendency = np.polyfit(np.array(range(len(uniformidad))), np.array(uniformidad), 1)[0]

                print("Uniformidad i: ", si)
                """
                plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
                """
            except:
                continue
            # pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
            # indexes = np.where(pvalues == pvalues.min())[0]
            # index = np.max(indexes)
            # # new samples, generate arbitrarily 1 or more at each iteration
            # goal_updated = float(np.random.uniform(bin_edges[index], bin_edges[index + 1], 1))
            # # distribution_goal = list(np.append(actual_distribution, goals_updated))
            # si = np.std(pvalues, axis=0)
            # print("Uniformidad i: ", si)
            # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)


        s = np.std(pvalues, axis=0)
        print("Uniformidad final: ", s)
        # print("Number of samples: ", len(r))
        """
        plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        """
        combinations_selected_i = sample_admin.get_combinations()
        pairs_selected.extend(combinations_selected_i)
        print('Combinations_selected: ', pairs_selected)
        len0, len2, len4, len6, len8 = sample_admin.get_overlap_lens()
        print('Index: ', index, 'lens: ', [len0, len2, len4, len6, len8])
        suma = suma + np.sum(np.array([len0, len2, len4, len6, len8]))
        print('Nº ejemplos selecciondados: ', suma, 'Nº ejemplos calculados: ', distance_overlap.get_len() - n_to_fill, 'Nº ejemplos desaprovechados: ', distance_overlap.get_len() - suma - n_to_fill)

    pairs_selected = np.unique(np.array(pairs_selected))  # Esta linea de aqui es la que hace que la distribucion no quede 100% uniforme
    return pairs_selected




def online_anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap, size):

    # delta_xy = 25  # metros
    # sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)

    # i = 150
    # sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
    # pairs_selected = get_online_pairs(sampled_positions, sampled_times, overlap)


    i = 185
    pairs_selected = []
    while len(pairs_selected) < size:
        sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
        pairs_selected = get_online_pairs(sampled_positions, sampled_times, overlap)
        print(len(pairs_selected))
        i += 1

    return pairs_selected


def online_anchor_uniform_distribution_ALL_INFO(positions, distances, reference_timestamps, other_timestamps, overlap, size):

    # delta_xy = 25  # metros
    # sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)

    # i = 150
    # sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
    # pairs_selected = get_online_pairs(sampled_positions, sampled_times, overlap)


    i = 185
    pairs_selected = []
    while len(pairs_selected) < size:
        sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
        pairs_selected = get_online_pairs_ALL_INFO(sampled_positions, sampled_times, overlap, distances)
        print(len(pairs_selected))
        i += 1

    return pairs_selected



def online_anchor_grid_ALL_INFO(positions, distances, reference_timestamps, other_timestamps, overlap, size):

    # delta_xy = 25  # metros
    # sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)

    # i = 150
    # sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
    # pairs_selected = get_online_pairs(sampled_positions, sampled_times, overlap)


    i = 185
    pairs_selected = []
    while len(pairs_selected) < size:
        sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
        pairs_selected = get_online_grid_ALL_INFO(sampled_positions, sampled_times, overlap, distances)
        print(len(pairs_selected))
        i += 1

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

class SampleAdministrator():
    def __init__(self):
        self.overlap00_02 = []
        self.overlap02_04 = []
        self.overlap04_06 = []
        self.overlap06_08 = []
        self.overlap08_10 = []
        self.valid_candidates = []
        self.rejected_candidates = []

    def save_overlap(self, overlap):
        if overlap <= 0.2:
            self.overlap00_02.append(overlap)
        elif overlap > 0.2 and overlap <= 0.4:
            self.overlap02_04.append(overlap)
        elif overlap > 0.4 and overlap <= 0.6:
            self.overlap04_06.append(overlap)
        elif overlap > 0.6 and overlap <= 0.8:
            self.overlap06_08.append(overlap)
        elif overlap > 0.8 and overlap <= 1.0:
            self.overlap08_10.append(overlap)

    def save_candidate(self, combination_ID):
        self.valid_candidates.append(combination_ID)

    def classify_overlap(self, overlap):
        value = None
        if overlap <= 0.2:
            value = 0
        elif overlap > 0.2 and overlap <= 0.4:
            value = 2
        elif overlap > 0.4 and overlap <= 0.6:
            value = 4
        elif overlap > 0.6 and overlap <= 0.8:
            value = 6
        elif overlap > 0.8 and overlap <= 1.0:
            value = 8

        return value

    def get_overlap_lens(self):
        return len(self.overlap00_02), len(self.overlap02_04), len(self.overlap04_06), len(self.overlap06_08), len(self.overlap08_10)

    def get_combinations(self):
        return self.valid_candidates

    def manage_overlap(self, candidate, combination_ID):
        len0, len2, len4, len6, len8 = self.get_overlap_lens()
        # min_len = np.min([len0, len2, len4, len6, len8])
        max_len = np.max([len0, len2, len4, len6, len8])
        category = self.classify_overlap(candidate)
        skip_to = None
        if category == 8:
            if len8 == max_len:
                self.save_overlap(candidate)
                self.valid_candidates.append(combination_ID)
            else:
                skip_to = 6
        elif category == 6:
            if len6 < max_len:
                self.save_overlap(candidate)
                self.valid_candidates.append(combination_ID)
                len6 = len(self.overlap06_08)
                if len6 == max_len:
                    skip_to = 4
            else:
                skip_to = 4
        elif category == 4:
            if len4 < max_len:
                self.save_overlap(candidate)
                self.valid_candidates.append(combination_ID)
                len4 = len(self.overlap04_06)
                if len4 == max_len:
                    skip_to = 2
            else:
                skip_to = 2
        elif category == 2:
            if len2 < max_len:
                self.save_overlap(candidate)
                self.valid_candidates.append(combination_ID)
                len2 = len(self.overlap02_04)
                if len2 == max_len:
                    skip_to = 0
            else:
                skip_to = 0
        elif category == 0:
            if len0 < max_len:
                self.save_overlap(candidate)
                self.valid_candidates.append(combination_ID)
                len0 = len(self.overlap00_02)
                if len0 == max_len:
                    skip_to = -1
            else:
                skip_to = -1

        min_len = np.min([len0, len2, len4, len6, len8])
        max_len = np.max([len0, len2, len4, len6, len8])
        if min_len != max_len and skip_to == -1:
            # lens = [len0, len2, len4, len6, len8]
            # index = lens.index(min(lens))
            # if index == 0:
            #     skip_to = 2
            skip_to = None

        return skip_to


class DistanceOverlap_Relation():
    def __init__(self):
        self.overlaps = []
        self.distances = []
        self.tendency = None
    def add(self, overlap, distance):
        self.overlaps.append(overlap)
        self.distances.append(distance)

    def get_len(self):
        return len(self.overlaps)

    def polyfit_with_fixed_points(self, n, x, y, xf, yf):
        """
        where n is the degree, points `(x, y)` of the polynomial, xf and yf the number of fixed points

          The mathematically correct way of doing a fit with fixed points is to use Lagrange multipliers.
        Basically, you modify the objective function you want to minimize, which is normally the sum of
        squares of the residuals, adding an extra parameter for every fixed point. I have not succeeded
        in feeding a modified objective function to one of scipy's minimizers. But for a polynomial fit,
        you can figure out the details with pen and paper and convert your problem into the solution of
        a linear system of equations
        """
        len_xf = 1
        mat = np.empty((n + 1 + len_xf,) * 2)
        vec = np.empty((n + 1 + len_xf,))
        x_n = x ** np.arange(2 * n + 1)[:, None]
        yx_n = np.sum(x_n[:n + 1] * y, axis=1)
        x_n = np.sum(x_n, axis=1)
        idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
        mat[:n + 1, :n + 1] = np.take(x_n, idx)
        xf_n = xf ** np.arange(n + 1)[:, None]
        mat[:n + 1, n + 1:] = xf_n / 2
        mat[n + 1:, :n + 1] = xf_n.T
        mat[n + 1:, n + 1:] = 0
        vec[:n + 1] = yx_n
        vec[n + 1:] = yf
        params = np.linalg.solve(mat, vec)
        return params[:n + 1]

    def get_polyline_with_fixed_points(self, deg):
        x = np.array(self.distances)
        y = np.array(self.overlaps)
        xf = 0.0
        yf = 1.0
        params = self.polyfit_with_fixed_points(deg, x, y, xf, yf)
        poly = np.polynomial.Polynomial(params)
        return poly


    def get_polyline(self, deg=4):
        poly = np.polyfit(x=np.array(self.distances), y=np.array(self.overlaps), deg=deg)
        return np.poly1d(poly)
    def predict_overlap(self, distance):
        # self.tendency = self.get_polyline(deg=10)
        self.tendency = self.get_polyline_with_fixed_points(deg=10)
        overlap_prediction = self.tendency(distance)
        return overlap_prediction

    def plot_tendency(self):
        p10 = self.get_polyline(deg=10)
        # p30 = self.get_polyline(deg=30)
        f10 = self.get_polyline_with_fixed_points(deg=10)
        xp = np.linspace(0, 350, 600)
        _ = plt.plot(np.array(self.distances), np.array(self.overlaps), '.', xp, p10(xp), '-', xp, f10(xp), '--')
        plt.ylabel('Overlap')
        plt.xlabel('Distance (m)')
        plt.show()

    def plot_occupancy_grid(self):

        # If you do not set the same values for X and Y, the bins won't be a square!
        max_distance = np.max(np.array(self.distances))
        bin_x = int(max_distance/5)
        h, x_edges, y_edges, image = plt.hist2d(np.array(self.distances), np.array(self.overlaps), bins=(bin_x, 10), cmap=plt.cm.Greys)
        plt.show()
        h_tras = h.T
        h_norm = np.divide(h_tras, np.amax(h_tras, axis=1).reshape(10, 1))
        X, Y = np.meshgrid(x_edges, y_edges)
        plt.pcolormesh(X, Y, h_norm, cmap=plt.cm.Greys)
        plt.show()

    def get_occupancy_grid(self):

        # If you do not set the same values for X and Y, the bins won't be a square!
        max_distance = np.max(np.array(self.distances))
        bin_x = int(max_distance/5)
        bin_y = 10
        # h, x_edges, y_edges, image = plt.hist2d(np.array(self.distances), np.array(self.overlaps), bins=(bin_x, 10), cmap=plt.cm.Greys)
        h, x_edges, y_edges = np.histogram2d(np.array(self.distances), np.array(self.overlaps), bins=(bin_x, np.linspace(0, 1, bin_y + 1)), density=True)

        h_tras = h.T
        h_norm = np.divide(h_tras, np.amax(h_tras, axis=1).reshape(bin_y, 1))
        # X, Y = np.meshgrid(x_edges, y_edges)
        # plt.pcolormesh(X, Y, h_norm, cmap=plt.cm.Greys)
        # plt.show()
        return h_norm, x_edges, y_edges

    def distances2search(self, overlap_value):
        h_norm, x_edges, y_edges = self.get_occupancy_grid()
        y_bin = np.digitize(overlap_value, y_edges) - 1 #get the bin corresponding to the overlap value
        # index_candidate = np.argmax(h_norm[y_bin])
        index_candidate = np.random.choice(np.arange(0, len(h_norm[y_bin])), 1, p=h_norm[y_bin]/np.sum(h_norm[y_bin]))
        min_distance = x_edges[index_candidate]
        max_distance = x_edges[index_candidate + 1]

        return min_distance, max_distance





class SampleStorage():
    def __init__(self):
        self.reference_timestamps = []
        self.other_timestamps = []
        self.saved_overlaps = []

    def add(self, ref_time, other_time, overlap):
        self.reference_timestamps.append(ref_time)
        self.other_timestamps.append(other_time)
        self.saved_overlaps.append(overlap)

def compute_distances(df):
    ref_x = np.array(df["Reference x"])
    ref_y = np.array(df["Reference y"])
    other_x = np.array(df["Other x"])
    other_y = np.array(df["Other y"])
    distances = []

    for i in range(0, len(ref_x)):
        dxy = np.linalg.norm(np.array([ref_x[i], ref_y[i]]) - np.array([other_x[i], other_y[i]]))
        distances.append(dxy)

    return np.array(distances)

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
    distances = compute_distances(df)



    pairs_selected_globally = global_uniform_distribution(overlap)
    print('Offline global uniform selection: ', len(pairs_selected_globally))

    # pairs_selected_online_anchor = online_anchor_uniform_distribution(positions, reference_timestamps, other_timestamps,
    #                                                                   overlap, size=len(pairs_selected_globally))
    #
    # print('Online selection len: ', len(pairs_selected_online_anchor))

    pairs_selected_online_grid_ALL_INFO = online_anchor_grid_ALL_INFO(positions, distances,
                                                                                        reference_timestamps,
                                                                                        other_timestamps,
                                                                                        overlap, size=len(
            pairs_selected_globally))

    print('Online grid selection ALL INFO len: ', len(pairs_selected_online_grid_ALL_INFO))


    pairs_selected_online_anchor_ALL_INFO = online_anchor_uniform_distribution_ALL_INFO(positions, distances, reference_timestamps, other_timestamps,
                                                                      overlap, size=len(pairs_selected_globally))

    print('Online selection ALL INFO len: ', len(pairs_selected_online_anchor_ALL_INFO))

    """
    pairs_selected_partial_anchor = anchor_partial_uniform_distribution(positions, reference_timestamps,
                                                                        other_timestamps, overlap, size=len(pairs_selected_globally))
    print('Offline partial uniform selection: ', len(pairs_selected_partial_anchor))
   
    # pairs_selected_anchor = anchor_uniform_distribution(positions, reference_timestamps, other_timestamps, overlap,
    #                                                     size=len(pairs_selected_globally))



    pairs_selected_randomly = random_distribution(overlap, size=len(pairs_selected_partial_anchor))
    """
    # Descomentar si se quiere crear el csv
    """
    write_csv(pairs_selected_globally, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='global_uniform')
    write_csv(pairs_selected_anchor, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='anchor_uniform')
    write_csv(pairs_selected_randomly, reference_timestamps, other_timestamps, overlap, reference_x, reference_y, other_x, other_y, name='random')
    """

    write_csv(pairs_selected_online_grid_ALL_INFO, reference_timestamps, other_timestamps, overlap, reference_x, reference_y,
              other_x, other_y, name='online_anchor_GRID_ALL_INFO')
    # print(len(pairs_selected_anchor))

    # print('Offline random selection', len(pairs_selected_randomly))
    print('Offline global uniform selection: ', len(pairs_selected_globally))
    # print('Offline partial uniform selection: ', len(pairs_selected_partial_anchor))
    # print('Online selection len: ', len(pairs_selected_online_anchor))




