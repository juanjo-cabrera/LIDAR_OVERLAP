from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS
import multiprocessing as mp
import os
from multiprocessing import get_context, set_start_method

scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
scan_indices = np.arange(0, len(scan_times))
scan_combinations = list(it.combinations(scan_indices, 2))

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


def worker_diff(queue_out, queue_in):
    data = []
    index = queue_in.get()
    print('Calculated: ', index, 'overlaps out of ', len(scan_combinations))
    idx_reference, idx_other = scan_combinations[index]
    overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
    # print('\nCalculated overlap: ', overlap,' idx_reference: ', idx_reference, ' idx_other: ', idx_other)
    local_data = [scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh,
     pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]]
    data.append(local_data)
    # print("-------------------------------sending-------------------------------")
    queue_out.put(data)

def worker_same(queue_out, queue_in):
    data = []
    idx = queue_in.get()
    data.append([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
    queue_out.put(data)

def listener(queue):
    i = 0
    with open(EXP_PARAMETERS.directory + '/multi_labelling.csv', 'w', newline='') as file:
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

def main():
    manager = mp.Manager()
    queue_out = manager.Queue()
    queue_in1 = manager.Queue()
    queue_in2 = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    watcher = pool.apply_async(listener, (queue_out,))

    for idx in scan_indices:
        queue_in1.put(idx)
        job = pool.apply_async(worker_same, args=(queue_out, queue_in1))

    for i in range(0, len(scan_combinations)):
        queue_in2.put(i)
        print('Indice diff: ', i)
        job = pool.apply_async(worker_diff, args=(queue_out, queue_in2))

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




