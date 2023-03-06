
import pandas as pd

import matplotlib.pyplot as plt


class PointnetVLAD_submap_reader():
    def __init__(self, directory):
        self.directory = directory

    def plot_submap(self):
        """Visualize the trajectory"""
        train_northing, train_easting, test_northing, test_easting = self.submap_data()

        # set up plot
        fig, ax = plt.subplots()

        # map poses
        ax.scatter(train_easting, train_northing, c='grey', s=10)
        ax.scatter(test_easting, test_northing, c='red', s=10)

        ax.axis('square')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Poses')
        plt.show()

    def submap_data(self):
        train_northing, train_easting = self.read_gps_data(train=True)
        test_northing, test_easting = self.read_gps_data(train=False)
        return train_northing, train_easting, test_northing, test_easting

    def read_gps_data(self, train):
        if train == True:
            csv_filename = self.directory + '/pointcloud_locations_20m_10overlap.csv'
        else:
            csv_filename = self.directory + '/pointcloud_locations_20m.csv'
        df = pd.read_csv(csv_filename)
        northing = df['northing']
        easting = df['easting']

        return northing, easting


if __name__ == "__main__":
    # Prepare data
    oxford_read = PointnetVLAD_submap_reader(directory='/home/arvc/Escritorio/develop/SparseConv/benchmark_datasets/oxford/2014-11-14-16-34-33')
    oxford_read2 = PointnetVLAD_submap_reader(directory='/home/arvc/Escritorio/develop/SparseConv/benchmark_datasets/oxford/2015-11-12-11-22-05')

    oxford_read.plot_submap()
    oxford_read2.plot_submap()