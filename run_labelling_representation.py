import pandas as pd
from config import TRAINING_PARAMETERS
from run_3D_overlap import *

class VisualizeLabels():
    def __init__(self, directory):
        self.root_dir = directory
        labels_dir = self.root_dir + '/all_combinations.csv'
        df_labels = pd.read_csv(labels_dir)

        self.reference_timestamps = np.array(df_labels["Reference timestamp"])
        self.other_timestamps = np.array(df_labels["Other timestamp"])
        self.overlap = np.array(df_labels["Overlap"])
        self.overlap_poses = np.array(df_labels["Overlap poses"])
        self.overlap_fpfh = np.array(df_labels["Overlap fpfh"])

        scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=self.root_dir)
        # indices = np.arange(0, len(scan_times))
        # indices = np.where(self.reference_timestamps == self.other_timestamps)
        self.timestamps = scan_times
        self.matrix_zeros = np.zeros((len(scan_times), len(scan_times)))
        self.matrix_ones = np.ones((len(scan_times), len(scan_times)))
        self.positions = pos
        self.distances = []

    def overlap_correlation(self):
        for idx in range(0, len(self.overlap)):
            ref_times = self.reference_timestamps[idx]
            other_times = self.other_timestamps[idx]
            ref_idx = np.where(self.timestamps == ref_times)
            other_idx = np.where(self.timestamps == other_times)
            self.matrix_zeros[ref_idx, other_idx] = self.overlap[idx]
            self.matrix_zeros[other_idx, ref_idx] = self.overlap[idx]
            self.matrix_ones[ref_idx, other_idx] = self.overlap[idx]
            self.matrix_ones[other_idx, ref_idx] = self.overlap[idx]

        self.plot_overlap_correlation(self.matrix_zeros)
        self.plot_overlap_correlation(self.matrix_ones)

    def overlap_vs_distance(self, which_overlap):
        if which_overlap == 'poses':
            overlap = self.overlap_poses
        elif which_overlap == 'fpfh':
            overlap = self.overlap_fpfh
        else:
            overlap = self.overlap
        self.distances = []
        for idx in range(0, len(overlap)):
            ref_times = self.reference_timestamps[idx]
            other_times = self.other_timestamps[idx]
            ref_idx = np.where(self.timestamps == ref_times)
            other_idx = np.where(self.timestamps == other_times)
            self.distances.append(np.linalg.norm(self.positions[ref_idx] - self.positions[other_idx]))

        self.plot_overlap_vs_disntance(which_overlap)


    def segmented_overlap_vs_distance(self):
        self.switch = np.empty(self.overlap.shape)
        self.distances = []
        for idx in range(0, len(self.overlap)):
            if self.overlap_poses[idx] >= self.overlap_fpfh[idx]:
                self.switch[idx] = 1

            else:
                self.switch[idx] = 0
            ref_times = self.reference_timestamps[idx]
            other_times = self.other_timestamps[idx]
            ref_idx = np.where(self.timestamps == ref_times)
            other_idx = np.where(self.timestamps == other_times)
            self.distances.append(np.linalg.norm(self.positions[ref_idx] - self.positions[other_idx]))

        self.distances = np.array(self.distances)
        self.plot_seg_overlap_distance()

    def plot_seg_overlap_distance(self):
        idx_poses = np.where(self.switch == 1)[0]
        idx_fpfh = np.where(self.switch == 0)[0]
        idx_failed_poses = np.where(self.switch == 0)[0]

        fig, ax = plt.subplots()
        ax.scatter(np.array(self.distances[idx_poses]), self.overlap[idx_poses], c='red', s=10)
        ax.scatter(np.array(self.distances[idx_fpfh]), self.overlap[idx_fpfh], c='blue', s=10)
        # ax.scatter(np.array(self.distances[idx_failed_poses]), self.overlap_poses[idx_failed_poses], c='green', s=10)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Overlap')
        ax.legend(['Poses', 'FPFH'])
        plt.show(block=False)

    def plot_overlap_vs_disntance(self, which_overlap):
        if which_overlap == 'poses':
            overlap = self.overlap_poses
        elif which_overlap == 'fpfh':
            overlap = self.overlap_fpfh
        else:
            overlap = self.overlap

        fig, ax = plt.subplots()
        ax.scatter(np.array(self.distances), overlap, c='grey', s=10)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Overlap')
        plt.show(block=False)

    def plot_overlap_correlation(self, matrix):
        plt.figure()
        plt.imshow(np.array(matrix, dtype=float), cmap='gray')
        plt.show(block=False)


if __name__ == '__main__':
    # vis = VisualizeLabels(directory=TRAINING_PARAMETERS.training_path)
    vis = VisualizeLabels(directory=EXP_PARAMETERS.directory)
    vis.overlap_correlation()
    vis.overlap_vs_distance('poses')
    vis.overlap_vs_distance('fpfh')
    vis.overlap_vs_distance('max')
    vis.segmented_overlap_vs_distance()

    plt.show()
