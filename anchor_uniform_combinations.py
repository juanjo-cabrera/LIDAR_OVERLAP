"""
In this script we assess the labelling of some combinations of a given trajectory in order ot have an uniform histogram of the overlap
"""
import random

import numpy as np

from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS, ICP_PARAMETERS
from scan_tools.keyframe import KeyFrame
from sklearn.neighbors import KDTree
from scipy import interpolate
import pandas as pd

def process_overlap(keyframe_manager, poses, scan_idx, i, dist):
    # pre_process = True

    current_pose = poses[scan_idx].array
    reference_pose = poses[i].array

    # if pre_process:
    #     keyframe_manager.keyframes[scan_idx].pre_process(plane_model=plane_model)
    #     keyframe_manager.keyframes[i].pre_process(plane_model=plane_model)

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


def get_combinationID(reference_time, other_time, reference_timestamps, other_timestamps):
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
        distances, indices = kd_tree.query(np.array([sampled_position]), k=len(all_positions))
        # indices = kd_tree.query_radius(np.array([sampled_position]), r=5)
        indices = np.array(list(indices))
        nearest_times = all_times[indices].flatten()  # para que me salga del tipo (10,)
        overlaps = []
        combinations_proposed = []
        for nearest_time in nearest_times:
            combination_proposed = [sampled_time, nearest_time]
            list_idx = get_combination(sampled_time, nearest_time, reference_timestamps, other_timestamps)
            try:
                overlap_s, combination_selected = get_overlap(sampled_time, nearest_time, reference_timestamps, other_timestamps, overlap)
                # overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other,
                #                                                       plane_model)

            except:
                continue





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

class SampleAdministrator():
    def __init__(self):
        self.overlap00_02 = []
        self.overlap02_04 = []
        self.overlap04_06 = []
        self.overlap06_08 = []
        self.overlap08_10 = []
        self.valid_candidates = []
        self.overlaps = []
        self.overlaps_pose = []
        self.overlaps_fpfh = []
        self.rejected_candidates = []

    # def save_overlap(self, overlap):
    #     if overlap <= 0.2:
    #         self.overlap00_02.append(overlap)
    #     elif overlap > 0.2 and overlap <= 0.4:
    #         self.overlap02_04.append(overlap)
    #     elif overlap > 0.4 and overlap <= 0.6:
    #         self.overlap04_06.append(overlap)
    #     elif overlap > 0.6 and overlap <= 0.8:
    #         self.overlap06_08.append(overlap)
    #     elif overlap > 0.8 and overlap <= 1.0:
    #         self.overlap08_10.append(overlap)

    def save_overlap(self, overlap, overlap_pose, overlap_fpfh):
        self.overlaps.append(overlap)
        self.overlaps_pose.append(overlap_pose)
        self.overlaps_fpfh.append(overlap_fpfh)

    def save_candidate(self, combination_ID):
        self.valid_candidates.append(combination_ID)

    def check_candidate(self, combination_ID):
        index = np.where(np.array(self.valid_candidates) == combination_ID)[0]
        if len(index) > 0:
            overlap = self.overlaps[index[0]]
        else:
            overlap = None
        return overlap

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
        return self.valid_candidates, self.overlaps, self.overlaps_pose, self.overlaps_fpfh

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

def fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances):

    for i in range(0, len(csv_distances)):
        distance = csv_distances[i]
        overlap = csv_overlap[i]
        distance_overlap.add(overlap, distance)

def get_combination(reference_time, other_time, reference_timestamps, other_timestamps, overlap):
    i_ref = np.where(reference_timestamps == reference_time)
    i_other = np.where(other_timestamps == other_time)
    match_i = np.intersect1d(i_ref, i_other)

    j_ref = np.where(reference_timestamps == other_time)
    j_other = np.where(other_timestamps == reference_time)
    match_j = np.intersect1d(j_ref, j_other)
    match = np.unique(np.concatenate((match_i, match_j)))
    return int(match)

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
                                                                                nearest_times[pair_candidate], distances[pair_candidate])
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
                                                                                    distances[pair_candidate])
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
def load_previous_knowledge(sequences):
    for sequence in sequences:
        df = pd.read_csv('/home/arvc/Juanjo/Datasets/KittiDataset/sequences/0' + str(sequence) + '/all_combinations.csv')
        csv_overlap = np.array(df["Overlap"])
        csv_distances = compute_distances(df)
        fill_ALL_predictor(distance_overlap, csv_overlap, csv_distances)

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


if __name__ == "__main__":

    distance_overlap = DistanceOverlap_Relation()
    sequences = [4]
    load_previous_knowledge(sequences)

    scan_times, poses, positions, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)


    #AÑADIR LO SIGUIENTE PARA PROCESAR EL PLANO DE TIERRA UNA SOLA VEZ
    # kf = KeyFrame(directory=EXP_PARAMETERS.directory, scan_time=random.choice(scan_times))
    # kf.load_pointcloud()
    # pointcloud_filtered = kf.filter_by_radius(ICP_PARAMETERS.min_distance, ICP_PARAMETERS.max_distance)
    # plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    #EN PROCESS_OVERLAP PASAR plane_model y este a PRE_PROCESS
    all_combinations, reference_timestamps, other_timestamps = get_all_possible_combinations(scan_times)

    pairs_selected, overlaps_selected, overlaps_pose, overlaps_fpfh = online_anchor_grid_ALL_INFO(positions)

    with open(EXP_PARAMETERS.directory + '/anchor_05_uniform.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])

        for i in pairs_selected:
            idx_reference = reference_timestamps[i]
            idx_other = other_timestamps[i]
            writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlaps_selected[i], overlaps_pose[i], overlaps_fpfh[i], positions[idx_reference, 0], positions[idx_reference, 1], positions[idx_other, 0], positions[idx_other, 1]])



