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
    reference_idx = reference_idx[1:len(reference_idx)] # Me quito el primero que se repite con other
    ref_row = np.repeat('ref', len(reference_idx)).T
    ref_labelled = np.vstack((reference_idx, ref_row))

    other_idx = np.where(reference_timestamp == other_timestamps)[0]
    other_row = np.repeat('other', len(other_idx)).T
    other_labelled = np.vstack((other_idx, other_row))

    idx_labelled = np.hstack((other_labelled, ref_labelled))

    overlap_idx = np.hstack((reference_idx, other_idx))
    overlap_idx = np.unique(overlap_idx)
    overlap = overlap[overlap_idx]

    poses = []
    for i in overlap_idx:
        idx = np.where(idx_labelled[0, :] == str(i))
        column = idx_labelled[1, idx]
        if column == 'ref':
            pose = other_poses[i]
        else:
            pose = reference_poses[i]
        poses.append(pose)

    poses = np.array(poses)
    plot_overlap(EXP_PARAMETERS.scan_idx, poses, overlap)


