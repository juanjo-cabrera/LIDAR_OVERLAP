import numpy as np
from tools.quaternion import Quaternion
from tools.conversions import rot2quaternion, rot2euler
import pandas as pd
from tools.homogeneousmatrix import HomogeneousMatrix
from pyproj import Proj
from config import EXP_PARAMETERS


class KittiReader():
    def __init__(self, directory):
        self.directory = directory

    def prepare_kitti_data(self, deltaxy, deltath, nmax_scans=None):
        print("PREPARING EXPERIMENT DATA FOR OUTDOOR EXPERIMENTS")
        # sample odometry at deltaxy and deltatheta

        sampled_times, pos, orient, poses = self.sample_poses(deltaxy=deltaxy, deltath=deltath)


        print("FOUND: ", len(sampled_times), "TOTAL SCANS")
       
        # return scan_times, odo_pos, odo_orient, gps_pos

        return sampled_times, pos, orient, poses

    def read_poses_data(self):
        poses_path = self.directory + '/poses.txt'
        # df = pd.read_csv(poses_filename, sep=" ", header=None)
        poses = []
        with open(poses_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                poses.append(T_w_cam0)

        return poses

    def read_calibration(self):
        """ Load calibrations (T_cam_velo) from file.
        """
        # Read and parse the calibrations
        calib_path = self.directory + '/calib.txt'
        T_cam_velo = []
        try:
            with open(calib_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Tr:' in line:
                        line = line.replace('Tr:', '')
                        T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                        T_cam_velo = T_cam_velo.reshape(3, 4)
                        T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

        except FileNotFoundError:
            print('Calibrations are not avaialble.')

        return np.array(T_cam_velo)

    def sample_poses(self, deltaxy=0.5, deltath=0.2):
        """
        Get pose times separated by dxy (m) and dth (rad)
        """

        sampled_times = []
        sampled_pos = []
        sampled_orient = []
        tranforms = []

        # load calibrations
        T_cam_velo = self.read_calibration()
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # load poses
        poses = self.read_poses_data()
        pose_scan_idx_inv = np.linalg.inv(poses[EXP_PARAMETERS.scan_idx])

        # for KITTI dataset, we need to convert the provided poses
        # from the camera coordinate system into the LiDAR coordinate system
        poses_new = []
        for pose in poses:
            poses_new.append(T_velo_cam.dot(pose_scan_idx_inv).dot(pose).dot(T_cam_velo))
        poses = np.array(poses_new)

        for ind in range(0, len(poses)):
            q = rot2quaternion(poses[ind])
            qw = q[0]
            qx = q[1]
            qy = q[2]
            qz = q[3]

            orientation = [qx, qy, qz, qw]
            q = Quaternion(q)
            position = poses[ind][0:3, 3]

            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])

            if ind == 0:
                sampled_times.append(ind)
                sampled_pos.append(position)
                sampled_orient.append(orientation)
                tranforms.append(HomogeneousMatrix(poses[ind]))
                odoi = odo
            odoi1 = odo

            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])

            if dxy > deltaxy or dth > deltath:
                sampled_times.append(ind)
                sampled_pos.append(position)
                sampled_orient.append(orientation)
                tranforms.append(HomogeneousMatrix(poses[ind]))
                odoi = odoi1
        return np.array(sampled_times), np.array(sampled_pos), np.array(sampled_orient), tranforms

    def get_closest_data(self, df_data, time_list, gps_mode='utm'):
        # df_odo = self.read_odometry_data()
        positions = []
        orientations = []
        corresp_time_list = []
        if gps_mode == 'utm':
            myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False, units='m')
        # now find odo corresponding to closest times
        for timestamp in time_list:
            # find the closest timestamp in df
            ind = df_data['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            try:
                position = [df_data['x'][ind], df_data['y'][ind], df_data['z'][ind]]
                positions.append(position)
            except:
                pass
            try:
                orientation = [df_data['qx'][ind], df_data['qy'][ind], df_data['qz'][ind], df_data['qw'][ind]]
                orientations.append(orientation)
            except:
                pass
            try:
                latitude = df_data['latitude'][ind]
                longitude = df_data['longitude'][ind]
                altitude = df_data['altitude'][ind]

                lat = np.array(latitude)
                lon = np.array(longitude)
                alt = np.array(altitude)
                if gps_mode == 'utm':
                    [x, y] = myProj(lon, lat)
                    position = [x, y, altitude]
                else:
                    position = [lat, lon, alt]
                positions.append(position)
            except:
                pass

            corresp_time = df_data['#timestamp [ns]'][ind]
            corresp_time_list.append(corresp_time)
        return np.array(corresp_time_list), np.array(positions), np.array(orientations)



if __name__ == "__main__":
    # Prepare data
    kitti_read = KittiReader(directory='/home/arvc/Escritorio/develop/Datasets/kitti_sample')

    scan_times, odom_ekf_pos, odom_ekf_orient, gps_pos = kitti_read.prepare_kitti_data(deltaxy=EXP_PARAMETERS.exp_deltaxy,
                                                                                     deltath=EXP_PARAMETERS.exp_deltath,
                                                                                     nmax_scans=EXP_PARAMETERS.exp_long)

