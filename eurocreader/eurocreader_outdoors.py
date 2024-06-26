import numpy as np
from tools.quaternion import Quaternion
import pandas as pd
from pyproj import Proj
from config import EXP_PARAMETERS


class EurocReader():
    def __init__(self, directory):
        self.directory = directory

    def prepare_experimental_data(self, deltaxy, deltath, nmax_scans=None, gps_mode='utm'):
        print("PREPARING EXPERIMENT DATA FOR OUTDOOR EXPERIMENTS")
        # eurocreader = EurocReader(directory=directory)
        # sample odometry at deltaxy and deltatheta
        odometry_times = self.sample_odometry(deltaxy=deltaxy, deltath=deltath)
        gps_times = self.filter_gps(EXP_PARAMETERS.gps_status)
        reference_times = self.get_common_times(odometry_times, gps_times)

        if nmax_scans is not None:
            print("CAUTION: CROPPING DATA TO: ", nmax_scans)
            odometry_times = odometry_times[0:nmax_scans]

        # read dfs from data

        df_odometry = self.read_odometry_data()
        df_scan_times = self.read_scan_times()
        df_gps = self.read_gps_data()
        # utm_x, utm_y = self.gps2utm(df_gps)
        # df_ground_truth = self.read_ground_truth_data()
        # for every time in odometry_times, find the closest times of a scan.
        # next, for every time of the scan, find again the closest odometry and the closest ground truth

        scan_times, _, _ = self.get_closest_data(df_scan_times, reference_times)
        _, odo_pos, odo_orient = self.get_closest_data(df_odometry, scan_times)
        _, gps_pos, _ = self.get_closest_data(df_gps, scan_times, gps_mode=gps_mode)
        # gps_pos = self.normalize_gps_data(gps_pos)
        # odo_pos = self.normalize_odom_data(odo_pos)


        print("FOUND: ", len(scan_times), "TOTAL SCANS")
        return scan_times, gps_pos, odo_pos, odo_orient

    def prepare_gps_data(self, deltaxy):
        print("PREPARING EXPERIMENT DATA FOR OUTDOOR EXPERIMENTS WITHOUT ODOMETRY")
        # sample odometry at deltaxy and deltatheta
        times_downsampled = self.sample_gps(deltaxy=deltaxy)
        times_filtered = self.filter_gps(EXP_PARAMETERS.gps_status)
        reference_times = self.get_common_times(times_downsampled, times_filtered)


        df_scan_times = self.read_scan_times()
        df_gps = self.read_gps_data()

        scan_times, _, _ = self.get_closest_data(df_scan_times, reference_times)
        # _, odo_pos, odo_orient = self.get_closest_data(df_odometry, scan_times)
        _, lat_lon, _ = self.get_closest_data(df_gps, scan_times, gps_mode='lat_lon')
        _, utm_pos, _ = self.get_closest_data(df_gps, scan_times, gps_mode='utm')


        print("FOUND: ", len(scan_times), "TOTAL SCANS")
        return scan_times, lat_lon, utm_pos


    def offline_ekf(self, odometry_times, df_odometry, scan_times):
        offset = odometry_times[0]

        scan_times = scan_times['#timestamp [ns]']
        odometry_times = odometry_times - offset
        odometry_times = odometry_times + scan_times[0]

        df_odometry['#timestamp [ns]'] = df_odometry['#timestamp [ns]'] - offset
        df_odometry['#timestamp [ns]'] = df_odometry['#timestamp [ns]'] + scan_times[0]

        return odometry_times, df_odometry

    def prepare_ekf_data(self, deltaxy, deltath, nmax_scans=None):
        print("PREPARING EXPERIMENT DATA FOR OUTDOOR EXPERIMENTS")
        # sample odometry at deltaxy and deltatheta
        odometry_times = self.sample_odometry(deltaxy=deltaxy, deltath=deltath)
        # gps_times = self.sample_gps(PARAMETERS.gps_status)
        # reference_times = self.get_common_times(odometry_times, gps_times)

        if nmax_scans is not None:
            print("CAUTION: CROPPING DATA TO: ", nmax_scans)
            odometry_times = odometry_times[0:nmax_scans]

        # read dfs from data
        df_odometry = self.read_odometry_data()
        df_scan_times = self.read_scan_times()
        df_gps = self.read_gps_data()
        # df_ground_truth = self.read_ground_truth_data()
        # for every time in odometry_times, find the closest times of a scan.
        # next, for every time of the scan, find again the closest odometry and the closest ground truth

        if EXP_PARAMETERS.do_offline_ekf:
            odometry_times, df_odometry = self.offline_ekf(odometry_times, df_odometry, df_scan_times)

        scan_times, _, _ = self.get_closest_data(df_scan_times, odometry_times)
        _, odo_pos, odo_orient = self.get_closest_data(df_odometry, scan_times)
        _, gps_pos, _ = self.get_closest_data(df_gps, scan_times, gps_mode='WGS84')
        # gps_pos = self.normalize_gps_data(gps_pos)
        # odo_pos = self.normalize_odom_data(odo_pos)

        print("FOUND: ", len(scan_times), "TOTAL SCANS")
        return scan_times, odo_pos, odo_orient, gps_pos

    def normalize_gps_data(self, gps_pos):
        gps_pos = gps_pos - gps_pos[0]
        return gps_pos

    def normalize_odom_data(self, odom_pos):
        odom_pos = odom_pos - odom_pos[0]
        return odom_pos

    def sample_gps(self, deltaxy=0.5):
        """
        Get odometry times separated by dxy (m) and dth (rad)
        """
        df = self.read_gps_data()
        UTMx, UTMy = self.gps2utm(df)
        odo_times = []
        for ind in df.index:
            odo = np.array([UTMx[ind], UTMy[ind]])
            current_time = df['#timestamp [ns]'][ind]
            if ind == 0:
                odo_times.append(current_time)
                odoi = odo
            odoi1 = odo

            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            if dxy > deltaxy:
                odo_times.append(current_time)
                odoi = odoi1
        return np.array(odo_times)

    def gps2utm(self, df_gps):
        latitude = df_gps['latitude']
        longitude = df_gps['longitude']
        status = df_gps['status']

        status_array = np.array(status)
        idx = np.where(status_array == 2)
        myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False, units='m')

        lat = np.array(latitude)
        lon = np.array(longitude)
        lat_ref = 38.275064630154375
        lon_ref = -0.6861944724090255

        UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)
        UTMx, UTMy = myProj(lon, lat)
        UTMx = UTMx[idx]
        UTMy = UTMy[idx]

        UTMx = UTMx - UTMx_ref
        UTMy = UTMy - UTMy_ref

        return UTMx, UTMy
    def read_gps_data(self):
        gps_csv_filename = self.directory + '/robot0/gps0/data.csv'
        df_gps = pd.read_csv(gps_csv_filename)
        return df_gps

    def filter_gps(self, reference_status):
        df_gps = self.read_gps_data()
        status = df_gps['status']
        timestamp = df_gps['#timestamp [ns]']

        status_array = np.array(status)
        idx = np.where(status_array == reference_status)
        timestamp = np.array(timestamp)
        timestamp = timestamp[idx]
        return timestamp


    def read_ekf_data(self):
        gt_csv_filename = self.directory + '/robot0/odometry_gps/data.csv'
        df_gt = pd.read_csv(gt_csv_filename)
        return df_gt

    def read_gps_filtered_data(self):
        gt_csv_filename = self.directory + '/robot0/gps_filtered/data.csv'
        df_gt = pd.read_csv(gt_csv_filename)
        return df_gt

    def read_ground_truth_data(self):
        gt_csv_filename = self.directory + '/robot0/ground_truth/data.csv'
        df_gt = pd.read_csv(gt_csv_filename)
        return df_gt

    def read_odometry_data(self):
        odo_csv_filename = self.directory + '/robot0/odom/data.csv'
        df_odo = pd.read_csv(odo_csv_filename)
        return df_odo

    def read_scan_times(self):
        scan_times_csv_filename = self.directory + '/robot0/lidar/data.csv'
        df_scan_times = pd.read_csv(scan_times_csv_filename)
        return df_scan_times


    def sample_odometry(self, deltaxy=0.5, deltath=0.2):
        """
        Get odometry times separated by dxy (m) and dth (rad)
        """
        df_odo = self.read_odometry_data()
        odo_times = []
        for ind in df_odo.index:
            # print(df_odo['x'][ind])
            position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
            q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])
            current_time = df_odo['#timestamp [ns]'][ind]
            if ind == 0:
                odo_times.append(current_time)
                odoi = odo
            odoi1 = odo

            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])
            if dxy > deltaxy or dth > deltath:
            # if dxy > deltaxy:
                odo_times.append(current_time)
                odoi = odoi1
        return np.array(odo_times)

    def get_common_times(self, odometry_times, gps_times):
        common_times = []

        for timestamp in gps_times:
            difference_times = odometry_times-timestamp
            result_index = np.argmin(abs(difference_times))
            common_times.append(odometry_times[result_index])

        common_times = np.unique(common_times)
        common_times = list(common_times)
        return common_times

    def get_closest_scan_times(self, odometry_times):
        df_scan_times = self.read_scan_times()
        scan_times = []
        # now find closest times to odo times
        for timestamp in odometry_times:
            result_index = df_scan_times['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            scan_times.append(df_scan_times['#timestamp [ns]'][result_index])
        return scan_times

    def get_closest_odometry(self, odometry_times):
        df_odo = self.read_odometry_data()
        odometry = []
        # now find odo corresponding to closest times
        for timestamp in odometry_times:
            ind = df_odo['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
            q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])
            odometry.append(odo)
        return odometry

    def get_closest_data(self, df_data, time_list, gps_mode='utm'):
        # df_odo = self.read_odometry_data()
        positions = []
        orientations = []
        corresp_time_list = []
        if gps_mode == 'utm':
            myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False, units='m')
            lat_ref = EXP_PARAMETERS.origin_lat
            lon_ref = EXP_PARAMETERS.origin_lon
            UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)

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
                    x = x - UTMx_ref
                    y = y - UTMy_ref
                    position = [x, y, 0]
                else:
                    position = [lat, lon, alt]
                positions.append(position)
            except:
                pass

            corresp_time = df_data['#timestamp [ns]'][ind]
            corresp_time_list.append(corresp_time)
        return np.array(corresp_time_list), np.array(positions), np.array(orientations)

