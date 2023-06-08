"""
The KeyFrameManager Stores a list of keyframes. Each keyframe is identified by a scan time and an index.

Methods provide a way to build a global map and save it to file.

"""
import numpy as np
from scan_tools.keyframe import KeyFrame
from tools.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
import random
from config import ICP_PARAMETERS

class KeyFrameManager():
    def __init__(self, directory, scan_times):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = scan_times
        self.keyframes = []

    def add_all_keyframes(self):
        for i in range(len(self.scan_times)):
            self.add_keyframe(i)

    def remove_keyframe(self, index):
        # kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index], index=index)

        del self.keyframes[index[0]: index[len(index) - 1] + 1]
        self.scan_times = np.delete(self.scan_times, index)
        # self.scan_times.pop(index)


    def load_pointclouds(self):
        read_failed = []
        for i in range(len(self.scan_times)):
            print('Loading pointcloud for keyframe: ', i, end='\r')
            success = self.keyframes[i].load_pointcloud()
            if success == False:
                message = 'Point cloud corresponding to scantime ' + str(self.scan_times[i]) + ' has no points'
                print(message)
                read_failed.append(int(i))

        if len(read_failed) != 0:
            self.remove_keyframe(read_failed)
        print('Ended loading poinclouds')
        print("SUCCESSFULLY READED: ", len(self.scan_times), "TOTAL SCANS")
        return self.scan_times

    def preprocessing_pointclouds(self):
        kf = KeyFrame(directory=self.directory, scan_time=random.choice(self.scan_times))
        kf.load_pointcloud()
        pointcloud_filtered = kf.filter_by_radius(ICP_PARAMETERS.min_distance, ICP_PARAMETERS.max_distance)
        plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

        for i in range(len(self.scan_times)):
            print('Preprocessing pointcloud for keyframe: ', i, end='\r')
            self.keyframes[i].pre_process(plane_model=plane_model)


    def add_keyframe(self, index):
        kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index])
        self.keyframes.append(kf)

    def save_solution(self, transforms):
        for i in range(len(transforms)):
            self.keyframes[i].transform = transforms[i]

    def set_relative_transforms(self, relative_transforms):
        """
        Given a set of relative transforms. Assign to each keyframe a global transform by
        postmultiplication.
        Caution, computing global transforms from relative transforms starting from T0=I
        """
        T = HomogeneousMatrix(np.eye(4))
        global_transforms = [T]
        for i in range(len(relative_transforms)):
            T = T*relative_transforms[i]
            global_transforms.append(T)

        for i in range(len(self.keyframes)):
            self.keyframes[i].set_global_transform(global_transforms[i])

    def set_global_transforms(self, global_transforms):
        """
        Assign the global transformation for each of the keyframes.
        """
        for i in range(len(global_transforms)):
            self.keyframes[i].set_global_transform(global_transforms[i])

    def compute_transformation_local_registration(self, i, j, method='A', initial_transform=np.eye(4)):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j when j-i = 1.
        An initial estimate is used to compute using icp
        """
        # compute initial transform from odometry
        # # TODO: Compute inintial transformation from IMU
        # if use_initial_transform:
        #     # initial estimation
        #     # xi = self.keyframes[i].x
        #     # xj = self.keyframes[j].x
        #     Ti = self.keyframes[i].transform # HomogeneousMatrix([xi[0], xi[1], 0], Euler([0, 0, xi[2]]))
        #     Tj = self.keyframes[j].transform # HomogeneousMatrix([xj[0], xj[1], 0], Euler([0, 0, xj[2]]))
        #     Tij = Ti.inv() * Tj
        #     # muatb = Tij.t2v()
        #     transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=Tij.array)
        #     atb = HomogeneousMatrix(transform.transformation)
        #     return atb
        # else:
        if method == 'A':
            atb, rmse = self.keyframes[i].local_registrationA(self.keyframes[j], initial_transform=initial_transform)
        elif method == 'B':
            atb, rmse = self.keyframes[i].local_registrationB(self.keyframes[j], initial_transform=initial_transform)
        elif method == 'point2point':
            atb, rmse = self.keyframes[i].local_registrationP2P(self.keyframes[j], initial_transform=initial_transform)

        return atb, rmse

    def compute_transformation_global_registration(self, i, j, method='FPFH'):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j.
        An initial estimate is used.
        FPFh to align and refine with icp
        Returning the ratio between the best and second best correlations.
        """
        if method == 'D':
            atb, prob = self.keyframes[i].global_registrationD(self.keyframes[j])
        elif method == 'FPFH':
            atb, prob = self.keyframes[i].global_registrationFPFH(self.keyframes[j])

        return atb, prob

    # def measure_centroid_difference(self, i, j):
    #     difference = self..keyframes[i].global_registrationD(self.keyframes[j])

    def compute_3d_overlap(self, i, j, atb):
        overlap = self.keyframes[i].pairwise_overlap(self.keyframes[j], transformation=atb)
        return overlap


    def compute_fpfh_similarity(self, i, j):
        similarity = self.keyframes[i].fpfh_similarity(self.keyframes[j])
        return similarity

    def compute_centroid(self, i, pcd_format):
        x, y = self.keyframes[i].pcd_centroid(pcd_format)


        # points = np.asarray(self.keyframes[i])
        # [x, y] = points[:, 0], points[:, 1]
        # x_mean = np.mean(x)
        # y_mean = np.mean(y)

        # x = x - x_mean
        # y = y - y_mean
        #
        # points[:, 0] = x
        # points[:, 1] = y

        # self.pointcloud_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # print(self.pointcloud_filtered.points)

        return x, y

    def compute_normalization(self, i):
        self.keyframes[i].normalize2center()

    def compute_normalization2maxbound(self, i):
        self.keyframes[i].normalize2maxbound()

    def compute_normalization2corner(self, i, corner):
        self.keyframes[i].normalize2corner(corner)

    def compute_normalization2minbound(self, i):
        self.keyframes[i].normalize2minbound()

    def compute_bbox(self, i):
        self.keyframes[i].obtain_corners()

    def view_map(self, keyframe_sampling=10, point_cloud_sampling=100):
        print("COMPUTING MAP FROM KEYFRAMES")
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(0, len(self.keyframes), keyframe_sampling):
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            # transform to global and
            pointcloud_temp = self.keyframes[i].transform_to_global(point_cloud_sampling=point_cloud_sampling)
            if pointcloud_temp is None:
                continue
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
        # draw the whole map
        o3d.visualization.draw_geometries([pointcloud_global])

    def save_to_file(self, filename, keyframe_sampling=10, point_cloud_sampling=100):
        print("COMPUTING MAP FROM KEYFRAMES")
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(0, len(self.keyframes), keyframe_sampling):
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            kf = self.keyframes[i]
            # transform to global and
            pointcloud_temp = kf.transform_to_global(point_cloud_sampling=point_cloud_sampling)
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
        print("SAVING MAP AS: ", filename)
        o3d.io.write_point_cloud(filename, pointcloud_global)
