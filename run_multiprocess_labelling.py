from run_3D_overlap import *
import itertools as it
import csv
from config import EXP_PARAMETERS
import multiprocessing as mp
import os
from multiprocessing import Process

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


def worker(indices):
    # data = []
    # scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
    # scan_indices = np.arange(0, len(scan_times))
    # scan_combinations = list(it.combinations(scan_indices, 2))

    # for idx in scan_indices:
    #     data.append([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
    #     if len(data) > 30:
    #         # print("sending")
    #         queue.put(data)
    #         data = []

    # for i in range(0, len(scan_combinations)):

    # np.random.seed((os.getpid() * int(time.time())) % 123456789)
    # index = np.random.rand(0, len(scan_combinations))

    # idx_reference, idx_other = scan_combinations.pop(0)

    # while len(scan_combinations) > 0:
    #     print('Combinations to compute: ', len(scan_combinations))
    #     # idx_reference = scan_combinations[i][0]
    #     # idx_other = scan_combinations[i][1]
    #     overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
    #     data.append([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh,
    #                      pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])
    #     if len(data) > 30:
    #         # print("sending")
    #         queue.put(data)
    #         data = []

    # print('Combinations to compute: ', len(scan_combinations))
    idx_reference = scan_combinations[indices][0]
    idx_other = scan_combinations[indices][1]
    print('\nidx_reference: ', idx_reference, ' idx_other: ', idx_other)
    overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
    # data.append([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh,
    #                  pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])
    # if len(data) > 30:
    #     # print("sending")
    #     queue.put(data)
    #     data = []




def listener(queue):
    i = 0
    with open(EXP_PARAMETERS.directory + '/labelling.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
        while 1:
            i += 1
            data = queue.get()
            writer.writerows(data)
            if i > 20:
                i = 0
                file.flush()
                # real_size_bytes = os.stat(path_csv).st_size / 2.6
                # real_size_gb = real_size_bytes / 1024 ** 3
                # print("Guardado")
                # print("De momento pesa {} GiB".format(real_size_gb))
                # if real_size_gb > 1.0:
                #     print("Quitting")
                #     f.close()
                #     quit()



def f(x):

    return x*x

def ff(indices):
    idx_reference = scan_combinations[indices][0]
    idx_other = scan_combinations[indices][1]
    print('\nidx_reference: ', idx_reference, ' idx_other: ', idx_other)
def main():
    # print(T_aleatoria())
    # must use Manager queue here, or will not work
    # manager = mp.Manager()
    # queue = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    # watcher = pool.apply_async(listener, (queue,))

    # for i in range(1):
    # for i in range(200):
        # job = pool.apply_async(worker, (queue,))

    # print(pool.map(f, range(len(scan_combinations))))
    pool.map(worker, range(len(scan_combinations)))

    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # now we are done, kill the listener
    # queue.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    print("Number of cpu : ", mp.cpu_count())
    main()


    # scan_times, poses, pos, keyframe_manager, lat, lon = reader_manager(directory=EXP_PARAMETERS.directory)
    # scan_indices = np.arange(0, len(scan_times))
    # scan_combinations = list(it.combinations(scan_indices, 2))
    #
    # with open(EXP_PARAMETERS.directory + '/labelling.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Reference timestamp", "Other timestamp", "Overlap", "Overlap poses", "Overlap fpfh", "Reference x", "Reference y", "Other x", "Other y"])
    #     for idx in scan_indices:
    #         writer.writerow([scan_times[idx], scan_times[idx], 1.0, 1.0, 1.0, pos[idx, 0], pos[idx, 1], pos[idx, 0], pos[idx, 1]])
    #     for i in range(0, len(scan_combinations)):
    #         print('Calculated: ', i, 'overlaps out of ', len(scan_combinations))
    #         idx_reference = scan_combinations[i][0]
    #         idx_other = scan_combinations[i][1]
    #         overlap, overlap_pose, overlap_fpfh = process_overlap(keyframe_manager, poses, idx_reference, idx_other)
    #         writer.writerow([scan_times[idx_reference], scan_times[idx_other], overlap, overlap_pose, overlap_fpfh, pos[idx_reference, 0], pos[idx_reference, 1], pos[idx_other, 0], pos[idx_other, 1]])



