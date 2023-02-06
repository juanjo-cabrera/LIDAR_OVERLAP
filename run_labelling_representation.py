import pandas as pd
from config import TRAINING_PARAMETERS
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt



class VisualizeLabels():
    def __init__(self, directory):
        self.root_dir = directory
        labels_dir = self.root_dir + '/labelling.csv'
        df_labels = pd.read_csv(labels_dir)

        self.reference_timestamps = np.array(df_labels["Reference timestamp"])
        self.other_timestamps = np.array(df_labels["Other timestamp"])
        self.overlap = np.array(df_labels["Overlap"])
        indices = np.where(self.reference_timestamps == self.other_timestamps)
        self.timestamps = self.reference_timestamps[indices]
        self.matrix = np.zeros((np.array(indices).size, np.array(indices).size))


    def overlap_correlation(self):
        for idx in range(0, len(self.overlap)):
            ref_times = self.reference_timestamps[idx]
            other_times = self.other_timestamps[idx]
            ref_idx = np.where(self.timestamps == ref_times)
            other_idx = np.where(self.timestamps == other_times)

            self.matrix[ref_idx, other_idx] = self.overlap[idx]
            self.matrix[other_idx, ref_idx] = self.overlap[idx]

        # df_cm = pd.DataFrame(array, range(2), range(2))
        self.plot_overlap_correlation()
        # df_cm = pd.DataFrame(self.matrix, index=self.timestamps, columns=self.timestamps)
        # # plt.figure(figsize=(10,7))
        # sn.set(font_scale=1.4)  # for label size
        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # plt.show()

    def plot_overlap_correlation(self):
        plt.figure()
        plt.imshow(np.array(self.matrix, dtype=float), cmap='gray')
        plt.show()




if __name__ == '__main__':
    # visualize_labels = VisualizeLabels(directory=TRAINING_PARAMETERS.training_path)
    vis = VisualizeLabels(directory=TRAINING_PARAMETERS.training_path)
    vis.overlap_correlation()
