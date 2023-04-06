"""
In this script we assess the labelling of some combinations of a given trajectory in order ot have an uniform histogram of the overlap
"""
import random
from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS, ICP_PARAMETERS
from scan_tools.keyframe import KeyFrame
from sklearn.neighbors import KDTree
from scipy import interpolate

def process_overlap(keyframe_manager, poses, scan_idx, i, plane_model):
    pre_process = True

    current_pose = poses[scan_idx].array
    reference_pose = poses[i].array

    if pre_process:
        keyframe_manager.keyframes[scan_idx].pre_process(plane_model=plane_model)
        keyframe_manager.keyframes[i].pre_process(plane_model=plane_model)

    transformation_matrix = np.linalg.inv(current_pose).dot(reference_pose)

    dist = np.linalg.norm(transformation_matrix[0:2, 3])

    if dist == 0:
        overlap = 1.0
        overlap_pose = - 1
        overlap_fpfh = - 1

    elif dist < 10:
        atb, rmse = keyframe_manager.compute_transformation_local_registration(scan_idx, i, method='point2point',
                                                                           initial_transform=transformation_matrix)
        overlap = keyframe_manager.compute_3d_overlap(scan_idx, i, atb)
        overlap_pose = overlap
        overlap_fpfh = - 1

    elif dist > 100:
        atb, rmse = keyframe_manager.compute_transformation_global_registration(scan_idx, i, method='FPFH')
        overlap = keyframe_manager.compute_3d_overlap(scan_idx, i, atb)
        overlap_fpfh = overlap
        overlap_pose = - 1

    else:
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
            pos_i = pos
        pos_1 = pos
        dxy = np.linalg.norm(pos_1[0:2]-pos_i[0:2])

        if dxy > deltaxy:
            sampled_times.append(current_time)
            sampled_pos.append(pos)
            pos_i = pos_1
    return np.array(sampled_times), np.array(sampled_pos)

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


def get_combination(reference_time, other_time, reference_timestamps, other_timestamps):
    i_ref = np.where(reference_timestamps == reference_time)
    i_other = np.where(other_timestamps == other_time)
    match_i = np.intersect1d(i_ref, i_other)

    j_ref = np.where(reference_timestamps == other_time)
    j_other = np.where(other_timestamps == reference_time)
    match_j = np.intersect1d(j_ref, j_other)
    match = np.unique(np.concatenate((match_i, match_j)))
    return int(match)


def get_total_uniform_pairs(sampled_positions, sampled_times, all_positions, all_times, all_combinations, reference_timestamps, other_timestamps):
    kd_tree = KDTree(all_positions)
    pairs_selected = []

    for index in range(0, len(sampled_positions)):
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        _, indices = kd_tree.query(np.array([sampled_position]), k=len(all_positions))
        # indices = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = all_times[indices].flatten()  # para que me salga del tipo (10,)
        overlaps = []
        combinations_proposed = []
        for nearest_time in nearest_times:
            combination_proposed = [sampled_time, nearest_time]
            list_idx = get_combination(sampled_time, nearest_time, reference_timestamps, other_timestamps)

            overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other,
                                                                  plane_model)



            all_combinations.pop(list_idx)
            combinations_proposed.append(combination_proposed)

    #
    #     if index == 0:
    #         pairs_selected_i = partial_uniform_distribution(np.array(overlaps), np.array(combinations_proposed))
    #         pairs_selected.extend(pairs_selected_i)
    #     else:
    #         common_combinations = np.intersect1d(pairs_selected, combinations_proposed)
    #         if len(common_combinations) > 0:
    #             indexes_to_conserve = np.where(common_combinations != combinations_proposed)
    #             combinations_proposed = np.array(combinations_proposed)[indexes_to_conserve[0].astype(int)]
    #             overlaps = np.array(overlaps)[indexes_to_conserve[0].astype(int)]
    #         pairs_selected_i = partial_uniform_distribution(np.array(overlaps), np.array(combinations_proposed))
    #         pairs_selected.extend(pairs_selected_i)
    #
    # return pairs_selected


def anchor_uniform_distribution(all_positions, all_times, all_combinations, reference_timestamps, other_timestamps, size=None):

    if size is None:
        print('En none')
        delta_xy = 10  # metros
        sampled_times, sampled_positions = downsample(all_positions, scan_times, delta_xy)
        get_total_uniform_pairs(sampled_positions, sampled_times, all_positions, all_times, all_combinations, reference_timestamps, other_timestamps)

    else:
        pairs_selected = []
        i = 150
        while len(pairs_selected) < size:
            sampled_positions, sampled_times = interpolate_positions(positions, scan_times, i)
            # pairs_selected = get_partial_uniform_pairs(sampled_positions, sampled_times)
            pairs_selected = get_total_uniform_pairs(sampled_positions, sampled_times)
            print(len(pairs_selected))
            i += 1

class samples_administrator():
    def __init__(self):
        self.overlap00_02 = []
        self.overlap02_04 = []
        self.overlap04_06 = []
        self.overlap06_08 = []
        self.overlap08_10 = []

    def classify_overlap(self, overlap):
        if overlap <= 0.2:
            self.overlap00_02.append(overlap)
        elif overlap > 0.2 and overlap <= 0.4:
            self.overlap02_04.append(overlap)
        elif overlap > 0.4 and overlap <= 0.6:
            self.overlap04_06.append(overlap)
        elif overlap > 0.6 and overlap <= 0.8:
            self.overlap06_08.append(overlap)
        elif overlap > 0.8 and overlap <= 0.1:
            self.overlap08_10.append(overlap)

    def sample_controler(self):







if __name__ == "__main__":
    scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)


    #AÑADIR LO SIGUIENTE PARA PROCESAR EL PLANO DE TIERRA UNA SOLA VEZ
    kf = KeyFrame(directory=EXP_PARAMETERS.directory, scan_time=random.choice(scan_times))
    kf.load_pointcloud()
    pointcloud_filtered = kf.filter_by_radius(ICP_PARAMETERS.min_distance, ICP_PARAMETERS.max_distance)
    plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    #EN PROCESS_OVERLAP PASAR plane_model y este a PRE_PROCESS

    scan_indices = np.arange(0, len(scan_times))
    all_combinations = list(it.combinations_with_replacement(scan_indices, 2))
    reference_timestamps = []
    other_timestamps = []

    for i in range(0, len(all_combinations)):
        idx_reference = all_combinations[i][0]
        idx_other = all_combinations[i][1]
        reference_timestamps.append(scan_times[idx_reference])
        other_timestamps.append(scan_times[idx_other])

    anchor_uniform_distribution(pos, scan_times, all_combinations, reference_timestamps, other_timestamps)




    #
    # with open(EXP_PARAMETERS.directory + '/labelling_prueba.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
    #     for idx in scan_indices:
    #         writer.writerow([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
    #     for i in range(0, len(scan_combinations)):
    #         print('Calculated: ', i, 'overlaps out of ', len(scan_combinations))
    #         idx_reference = scan_combinations[i][0]
    #         idx_other = scan_combinations[i][1]
    #         overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other, plane_model)
    #         writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh, pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])
    #
    #

