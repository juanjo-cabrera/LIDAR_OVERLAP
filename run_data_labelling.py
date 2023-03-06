"""
In this script we assess the labelling of all the possible combinations of a given trajectory
"""


from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS

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


if __name__ == "__main__":
    scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
    scan_indices = np.arange(0, len(scan_times))
    scan_combinations = list(it.combinations(scan_indices, 2))

    with open(EXP_PARAMETERS.directory + '/labelling_prueba.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
        for idx in scan_indices:
            writer.writerow([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
        for i in range(0, len(scan_combinations)):
            print('Calculated: ', i, 'overlaps out of ', len(scan_combinations))
            idx_reference = scan_combinations[i][0]
            idx_other = scan_combinations[i][1]
            overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
            writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh, pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])



