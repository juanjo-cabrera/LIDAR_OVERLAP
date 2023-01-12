from run_3D_overlap import *
import pandas as pd
from config import EXP_PARAMETERS

def read_labelled_data():
    csv_filename = EXP_PARAMETERS.directory + '/labelling.csv'
    df = pd.read_csv(csv_filename)
    reference_timestamps = np.array(df["Reference timestamp"])
    other_timestamps = np.array(df["Other timestamp"])
    overlap = np.array(df["Overlap"])
    reference_x = np.array(df["Reference x"])
    reference_y = np.array(df["Reference y"])
    other_x = np.array(df["Other x"])
    other_y = np.array(df["Other y"])

    reference_poses = np.array([reference_x, reference_y]).T
    other_poses = np.array([other_x, other_y]).T

    return reference_timestamps, other_timestamps, overlap, reference_poses, other_poses

if __name__ == "__main__":
    reference_timestamps, other_timestamps, overlap, reference_poses, other_poses = read_labelled_data()
    reference_timestamp = reference_timestamps[EXP_PARAMETERS.scan_idx]
    reference_idx = np.where(reference_timestamp == reference_timestamps)[0]
    other_poses = other_poses[reference_idx]
    other_idx = np.where(reference_timestamp == other_timestamps)[0]
    reference_poses = reference_poses[other_idx]
    result_poses = np.vstack((other_poses, reference_poses))

    overlap_idx = np.unique(reference_idx, other_idx)
    result_poses = np.unique(result_poses, axis=0)
    overlap = overlap[overlap_idx]

    plot_overlap(EXP_PARAMETERS.scan_idx, result_poses, overlap)


