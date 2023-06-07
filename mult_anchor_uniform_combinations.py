"""
In this script we assess the labelling of some combinations in order to have a uniform histogram of overlap, but in this case with multiprocessing
"""

from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS, ICP_PARAMETERS
import multiprocessing as mp
from multiprocessing import set_start_method
import random
from scan_tools.keyframe import KeyFrame
from anchor_uniform_combinations import DistanceOverlap_Relation, load_previous_knowledge, get_all_possible_combinations, get_combinationID, SampleAdministrator
from sklearn.neighbors import KDTree
import pandas as pd

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

def fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances):
    for i in range(0, len(csv_distances)):
        distance = csv_distances[i]
        overlap = csv_overlap[i]
        distance_overlap.add(overlap, distance)

def load_previous_knowledge(sequences):
    for sequence in sequences:
        df = pd.read_csv('/home/arvc/Juanjo/Datasets/KittiDataset/sequences/0' + str(sequence) + '/all_combinations.csv')
        csv_overlap = np.array(df["Overlap"])
        csv_distances = compute_distances(df)
        fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances)


def process_overlap(keyframe_manager, poses, scan_idx, i, plane_model, dist):
    pre_process = True

    current_pose = poses[scan_idx].array
    reference_pose = poses[i].array

    if pre_process:
        keyframe_manager.keyframes[scan_idx].pre_process(plane_model=plane_model)
        keyframe_manager.keyframes[i].pre_process(plane_model=plane_model)

    transformation_matrix = np.linalg.inv(current_pose).dot(reference_pose)

    # dist = np.linalg.norm(transformation_matrix[0:2, 3])

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


def get_all_possible_combinations(scan_times):
    scan_indices = np.arange(0, len(scan_times))
    all_combinations = list(it.combinations_with_replacement(scan_indices, 2))
    reference_timestamps = []
    other_timestamps = []

    for i in range(0, len(all_combinations)):
        idx_reference = all_combinations[i][0]
        idx_other = all_combinations[i][1]
        reference_timestamps.append(scan_times[idx_reference])
        other_timestamps.append(scan_times[idx_other])
    return all_combinations, reference_timestamps, other_timestamps


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


def get_online_grid_ALL_INFO(sampled_positions, sampled_times, distance_overlap):
    print('GRID METHOD')
    kd_tree = KDTree(positions)
    pairs_selected = []
    overlaps_selected = []
    overlaps_fpfh = []
    overlaps_pose = []


    for index in range(0, len(sampled_positions)):
        print('Anchor ', index, ' out of ', len(sampled_positions))
        sampled_position = sampled_positions[index]
        sampled_time = sampled_times[index]
        distances, indices = kd_tree.query(np.array([sampled_position]), k=len(positions))
        distances = distances.flatten()
        indices = np.array(list(indices))
        nearest_times = scan_times[indices].flatten()  # para que me salga del tipo (10,)
        actual_distribution = []
        sample_admin = SampleAdministrator()

        """
        distance_overlap.plot_tendency()
        distance_overlap.plot_occupancy_grid()

        """

        N = 5  # number of bins
        pairs = np.where(distances <= 2)[0]
        for pair_candidate in pairs:

            combination_proposed = get_combinationID(sampled_time, nearest_times[pair_candidate],
                                                                  reference_timestamps, other_timestamps)
            overlap_calculated = sample_admin.check_candidate(combination_proposed)
            if overlap_calculated is None:
                overlap_candidate, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, sampled_time,
                                                                                nearest_times[pair_candidate], plane_model, distances[pair_candidate])
            else:
                overlap_candidate = overlap_calculated
                overlap_pose = -1
                overlap_fpfh = -1

            distance_overlap.add(overlap_candidate, distances[pair_candidate])
            sample_admin.save_overlap(overlap_candidate, overlap_pose, overlap_fpfh)
            sample_admin.save_candidate(combination_proposed)

            actual_distribution.append(overlap_candidate)
            distances = np.delete(distances, pair_candidate)
            nearest_times = np.delete(nearest_times, pair_candidate)

        uniformidad = []
        pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
        pvalues = pvalues * len(actual_distribution) / np.sum(pvalues)
        """
        # plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        """
        min_value_initial = pvalues[N - 1].min()
        min_value = pvalues.min()
        si = np.std(pvalues, axis=0)
        uniformidad.append(si)

        tendency = -1

        while tendency < 0 or round(min_value) != round(min_value_initial):
            if round(min_value) == round(min_value_initial):
                break

            indexes = np.where(pvalues == pvalues.min())[0]
            index = np.max(indexes)
            # new samples, generate arbitrarily 1 or more at each iteration

            goal_updated = float(np.random.uniform(bin_edges[index], bin_edges[index + 1], 1))
            min_distance, max_distance = distance_overlap.distances2search(goal_updated)
            pairs_candidate = np.where(np.bitwise_and(distances < max_distance, distances > min_distance))[0]
            try:
                pair_candidate = np.random.choice(pairs_candidate)

                combination_proposed = get_combinationID(sampled_time, nearest_times[pair_candidate],
                                                         reference_timestamps, other_timestamps)
                overlap = sample_admin.check_candidate(combination_proposed)
                if overlap is None:
                    overlap_candidate, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses,
                                                                                    sampled_time,
                                                                                    nearest_times[pair_candidate],
                                                                                    plane_model, distances[pair_candidate])
                else:
                    overlap_candidate = overlap
                    overlap_pose = -1
                    overlap_fpfh = -1

                distance_overlap.add(overlap_candidate, distances[pair_candidate])
                # print('reality: ', overlap_candidate)
                sample_admin.save_overlap(overlap_candidate, overlap_pose, overlap_fpfh)
                sample_admin.save_candidate(combination_proposed)

                actual_distribution.append(overlap_candidate)
                distances = np.delete(distances, pair_candidate)
                nearest_times = np.delete(nearest_times, pair_candidate)

                pvalues, bin_edges = np.histogram(actual_distribution, bins=np.linspace(0, 1, N + 1), density=True)
                pvalues = pvalues * len(actual_distribution) / np.sum(pvalues)
                min_value = pvalues.min()

                si = np.std(pvalues, axis=0)
                uniformidad.append(si)
                tendency = np.polyfit(np.array(range(len(uniformidad))), np.array(uniformidad), 1)[0]

                """
                plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
                """
            except:
                continue
        """
        plot_hist(bin_edges=bin_edges, pvalues=pvalues, width=1 / N)
        """
        combinations_selected_i, overlaps_i, overlaps_pose_i, overlaps_fpfh_i = sample_admin.get_combinations()
        pairs_selected.extend(combinations_selected_i)
        overlaps_selected.extend(overlaps_i)
        overlaps_pose.extend(overlaps_pose_i)
        overlaps_fpfh.extend(overlaps_fpfh_i)

    pairs_selected, unique_indexes = np.unique(
        np.array(pairs_selected), return_index=True) # Esta linea de aqui es la que hace que la distribucion no quede 100% uniforme
    overlaps_selected = overlaps_selected[unique_indexes]
    overlaps_pose = overlaps_pose[unique_indexes]
    overlaps_fpfh = overlaps_fpfh[unique_indexes]
    return pairs_selected, overlaps_selected, overlaps_pose, overlaps_fpfh


def online_anchor_grid_ALL_INFO(positions):
    delta_xy = 1  # metros
    sampled_times, sampled_positions = downsample(positions, scan_times, delta_xy)
    pairs_selected, overlaps_selected, overlaps_pose, overlaps_fpfh = get_online_grid_ALL_INFO(sampled_positions, sampled_times, distance_overlap)
    print('EJEMPLOS SELECCIONADOS: -------------------------    ', len(pairs_selected))
    return pairs_selected, overlaps_selected, overlaps_pose, overlaps_fpfh


def worker():
    pairs_selected, overlaps_selected, overlaps_pose, overlaps_fpfh = online_anchor_grid_ALL_INFO(positions)
    with open(EXP_PARAMETERS.directory + '/mult_anchor_uniform.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x",
             "Reference y", "Other x", "Other y"])

        for i in pairs_selected:
            idx_reference = reference_timestamps[i]
            idx_other = other_timestamps[i]
            writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlaps_selected[i], overlaps_pose[i],
                             overlaps_fpfh[i], positions[idx_reference, 0], positions[idx_reference, 1],
                             positions[idx_other, 0], positions[idx_other, 1]])

def listener(queue):
    i = 0
    with open(EXP_PARAMETERS.directory + '/prueba.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
        while 1:
            i += 1
            data = queue.get()
            writer.writerows(data)
            print(data)
            if i > 20:
                i = 0
                file.flush()



distance_overlap = DistanceOverlap_Relation()
sequences = [4]
load_previous_knowledge(sequences)
scan_times, poses, positions, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
scan_indices = np.arange(0, len(scan_times))
scan_combinations = list(it.combinations(scan_indices, 2))

# AÑADIR LO SIGUIENTE PARA PROCESAR EL PLANO DE TIERRA UNA SOLA VEZ
kf = KeyFrame(directory=EXP_PARAMETERS.directory, scan_time=random.choice(scan_times))
kf.load_pointcloud()
pointcloud_filtered = kf.filter_by_radius(ICP_PARAMETERS.min_distance, ICP_PARAMETERS.max_distance)
plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

all_combinations, reference_timestamps, other_timestamps = get_all_possible_combinations(scan_times)
# EN PROCESS_OVERLAP PASAR plane_model y este a PRE_PROCESS
print('PROCESANDO SECUENCIA EN DIRECTORIO: ', EXP_PARAMETERS.directory)


def main():
    manager = mp.Manager()
    pool = mp.Pool(mp.cpu_count() + 2)
    job = pool.apply_async(worker, args=())
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # now we are done, kill the listener
    pool.close()
    pool.join()


if __name__ == "__main__":
    set_start_method('spawn') # spawn, fork (default on Unix), forkserver
    print("Number of cpu : ", mp.cpu_count())
    main()




