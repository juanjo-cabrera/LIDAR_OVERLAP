"""
    A Keyframe stores the pointcloud corresponding to a timestamp.
    The class includes methods to register consecutive pointclouds (local registration) and, as well, other pointclouds
    that may be found far away.
https://github.com/hankyang94/teaser_fpfh_threedmatch_python
"""
import numpy as np
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import open3d as o3d
import copy
from config import ICP_PARAMETERS, DEBUGGING_PARAMETERS, EXP_PARAMETERS, TRAINING_PARAMETERS


class KeyFrame():
    def __init__(self, directory, scan_time):
        # the estimated transform of this keyframe with respect to global coordinates
        self.transform = None
        # max radius to filter points
        self.max_radius = ICP_PARAMETERS.max_distance
        self.min_radius = ICP_PARAMETERS.min_distance
        # voxel sizes
        self.voxel_downsample_size = ICP_PARAMETERS.voxel_size # None
        self.voxel_size_normals = ICP_PARAMETERS.radius_normals
        self.voxel_size_normals_ground_plane = ICP_PARAMETERS.radius_gd
        # self.voxel_size_fpfh = 3*self.voxel_s
        self.icp_threshold = ICP_PARAMETERS.distance_threshold
        self.fpfh_threshold = ICP_PARAMETERS.fpfh_threshold
        # crop point cloud to this bounding box
        # self.dims_bbox = [40, 40, 40]
        # self.index = index
        self.timestamp = scan_time
        if directory.find('Kitti') == -1:
            self.filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'

        else:
            self.filename = directory + '/velodyne/' + '{:06d}'.format(scan_time) + '.bin'
        # all points
        self.pointcloud = None
        self.pointcloud_normalized = None
        # pcd with a segmented ground plate
        self.pointcloud_filtered = None
        self.pointcloud_ground_plane = None
        self.pointcloud_non_ground_plane = None
        self.pointcloud_training = None
        self.pcd_fpfh = None
        self.compute_groundplane_once = True
        self.plane_model = None



        # save the pointcloud for Scan context description

    def convert_kitti_bin_to_pcd(self, binFilePath):
        import struct
        size_float = 4
        list_pcd = []
        with open(binFilePath, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        return pcd
    def load_pointcloud(self):
        if self.filename.find('pcd') == -1:
            pointcloud = self.convert_kitti_bin_to_pcd(self.filename)
        else:
            pointcloud = o3d.io.read_point_cloud(self.filename)
        success = False
        if len(np.asarray(pointcloud.points)) != 0:
            self.pointcloud = pointcloud
            success = True
        return success

    def pre_process(self, plane_model=None):
        # bbox = self.dims_bbox
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-bbox[0], -bbox[1], -bbox[2]), max_bound=(bbox[0],
        #                                                                                                 bbox[1],
        #                                                                                                 bbox[2]))
        # self.pointcloud = self.pointcloud.crop(bbox)
        # self.draw_pointcloud()
        # filter by a max radius to avoid erros in normal computation
        self.pointcloud_filtered = self.filter_by_radius(self.min_radius, self.max_radius)
        self.pointcloud_filtered, ind = self.pointcloud_filtered.remove_radius_outlier(nb_points=3, radius=0.3)
        # inlier_cloud = cl.select_by_index(ind)
        # self.pointcloud_normalized = self.normalize2center()
        # self.draw_pointcloud(self.pointcloud_filtered)
        # downsample pointcloud and save to pointcloud in keyframe
        if self.voxel_downsample_size is not None:
            self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_downsample_size)

        # segment ground plane
        if plane_model is None:
            self.plane_model = self.calculate_plane(pcd=self.pointcloud_filtered)
        else:
            self.plane_model = plane_model

        pcd_ground_plane, pcd_non_ground_plane = self.segment_plane(self.plane_model, pcd=self.pointcloud_filtered)
        # _, self.pointcloud_training = self.segment_plane(self.pointcloud)
        self.pointcloud_ground_plane = pcd_ground_plane
        self.pointcloud_non_ground_plane = pcd_non_ground_plane
        # self.draw_pointcloud(self.pointcloud_non_ground_plane)
        # calcular las normales a cada punto
        # self.pointcloud_filtered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
        #                                                                       max_nn=ICP_PARAMETERS.max_nn))
        self.pointcloud_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
                                                                              max_nn=ICP_PARAMETERS.max_nn_gd))
        self.pointcloud_non_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=ICP_PARAMETERS.max_nn))
        self.pcd_fpfh = self.estimate_fpfh(radius=self.voxel_size_normals * 5, max_nn=100)

    def training_preprocess(self, plane_model):
        self.pointcloud_filtered = self.filter_by_radius(TRAINING_PARAMETERS.min_radius, TRAINING_PARAMETERS.max_radius)
        # self.pointcloud_normalized = self.normalize2center()
        # self.draw_pointcloud()
        # downsample pointcloud and save to pointcloud in keyframe
        if self.voxel_downsample_size is not None:
            # self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_downsample_size)
            self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=TRAINING_PARAMETERS.voxel_size)

        # segment ground plane
        _, pcd_non_ground_plane = self.segment_plane(plane_model, pcd=self.pointcloud_filtered)
        self.pointcloud_non_ground_plane = pcd_non_ground_plane
        if TRAINING_PARAMETERS.sample_points:
            pcd = self.fix_points_number(TRAINING_PARAMETERS.number_of_points)

        if TRAINING_PARAMETERS.normalize_coords:
            # pcd_features = self.local_normalize(self.pointcloud_non_ground_plane)
            pcd_features = self.global_normalize(self.pointcloud_non_ground_plane)
        else:
            pcd_features = self.pointcloud_non_ground_plane
        # features = np.ones(len(self.pointcloud_non_ground_plane.points))
        # features = features.reshape(len(features), 1)
        # features = self.extract_features(self.pointcloud_non_ground_plane)
        return np.asarray(self.pointcloud_non_ground_plane.points), np.asarray(pcd_features.points)

        # return np.asarray(self.pointcloud_non_ground_plane.points), features


    def estimate_fpfh(self, radius, max_nn):
        # radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud_non_ground_plane, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        return pcd_fpfh
    def points2homogeneous(self, pre_process):
        """
        Returns a nx4 numpy array of homogeneous points (x, y, z, 1).
        """
        if pre_process==True:
            points = np.asarray(self.pointcloud_non_ground_plane.points)
        else:
            points = np.asarray(self.pointcloud.points)

        # else:
        #     points = np.asarray(self.pointcloud.points)
        homogeneous_points = np.ones((points.shape[0], points.shape[1] + 1)) #creo una matriz de 1 con el tamaño de los ptos más una columna
        homogeneous_points[:, 0:3] = points #points mas la columna de 1
        return homogeneous_points

    def filter_by_radius(self, min_radius, max_radius):
        points = np.asarray(self.pointcloud.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        r2 = x**2 + y**2
        idx = np.where(r2 < max_radius**2) and np.where(r2 > min_radius ** 2)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))

    def normalize2center(self):

        self.pointcloud_normalized = copy.deepcopy(self.pointcloud_non_ground_plane)
        self.pointcloud_normalized = self.pointcloud_normalized.voxel_down_sample(voxel_size=0.2)

        points = np.asarray(self.pointcloud_normalized.points)
        self.point = copy.deepcopy(self.pointcloud_non_ground_plane)


        [x, y] = points[:, 0], points[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x = x - x_mean
        y = y - y_mean

        points[:, 0] = x
        points[:, 1] = y

        point_mean = np.array([0, 0, 0]).reshape(1, 3)
        self.point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_mean))

        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # print(self.pointcloud_filtered.points)

        # return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def local_normalize(self, pcd):
        """
        Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
        """
        pcd = copy.deepcopy(pcd)
        # points = np.asarray(self.pointcloud_normalized.points)
        points = np.asarray(pcd.points)

        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        x = x - x_mean
        y = y - y_mean
        z = z - z_mean

        x_max = np.max(x)
        y_max = np.max(y)
        z_max = np.max(z)
        x_min = np.abs(np.min(x))
        y_min = np.abs(np.min(y))
        z_min = np.abs(np.min(z))
        max_value = np.max([x_max, y_max, z_max, x_min, y_min, z_min])

        x = x / max_value
        y = y / max_value
        z = z / max_value

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return self.pointcloud_normalized


    def extract_features(self, pcd):
        """
        Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
        """

        points = np.asarray(pcd.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]

        z_max = np.max(z)
        z = z / z_max

        distance = np.sqrt(x ** 2 + y ** 2)
        max_distance = np.max(distance)
        distance = distance / max_distance

        features = np.concatenate((z.reshape(len(z), 1), distance.reshape(len(distance), 1)), axis=1)
        return features
    def global_normalize(self, pcd):
        """
        Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
        """
        pcd = copy.deepcopy(pcd)
        # points = np.asarray(self.pointcloud_normalized.points)
        points = np.asarray(pcd.points)

        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        x = x - x_mean
        y = y - y_mean
        z = z - z_mean

        x = x / TRAINING_PARAMETERS.max_radius
        y = y / TRAINING_PARAMETERS.max_radius
        z = z / TRAINING_PARAMETERS.max_radius

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return self.pointcloud_normalized



    def fix_points_number(self, number_neighbours):
        """
        Return a pcd filtered to the k points idx more close to the scan center
        """
        pcd = self.pointcloud_non_ground_plane
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        center_point = np.array([0, 0, 0]).reshape(1, 3)
        center_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(center_point))

        [k, idx, _] = pcd_tree.search_knn_vector_3d(center_point.points[0], number_neighbours)
        # np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

        return pcd.select_by_index(idx)



    def normalize2maxbound(self):
        self.pointcloud_normalized = copy.deepcopy(self.pointcloud_non_ground_plane)
        points = np.asarray(self.pointcloud_normalized.points)
        self.point = copy.deepcopy(self.pointcloud_non_ground_plane)


        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]

        [x_max, y_max, z_max] = self.pointcloud_normalized.get_max_bound()
        # max_bound = self.pointcloud_filtered.get_max_bound()
        # x_max = np.max(x)
        # y_max = np.max(y)
        # z_max = np.max(z)

        # point_max = np.array([x_max, y_max, z_max]).reshape(1, 3)
        point_max = np.array([0, 0, 0]).reshape(1, 3)
        self.point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_max))

        x = x - x_max
        y = y - y_max
        z = z - z_max

        points[:, 0] = x
        points[:, 1] = y
        # points[:, 2] = z

        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def obtain_corners(self):
        self.points_bbox = copy.deepcopy(self.pointcloud_non_ground_plane)

        bbox = self.pointcloud_normalized.get_oriented_bounding_box()

        corners_bbox = bbox.get_box_points()
        corners_bbox = np.asarray(corners_bbox)
        self.points_bbox = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners_bbox))

    def normalize2corner(self, corner):
        self.point = copy.deepcopy(self.pointcloud_non_ground_plane)
        self.pointcloud_normalized = copy.deepcopy(self.pointcloud_non_ground_plane)
        points = np.asarray(self.pointcloud_normalized.points)
        self.points_bbox = copy.deepcopy(self.pointcloud_non_ground_plane)

        bbox = self.pointcloud_normalized.get_oriented_bounding_box()

        corners_bbox = bbox.get_box_points()
        corners_bbox = np.asarray(corners_bbox)
        indices = np.where(corners_bbox[:, 2] < 0)
        corners_bbox = corners_bbox[indices, :]
        corners_bbox = corners_bbox.reshape(4, 3)

        # [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        self.point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([0, 0, 0]).reshape(1, 3)))

        points = points - corners_bbox[corner]
        corners_bbox = corners_bbox - corners_bbox[corner]

        self.points_bbox = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners_bbox))
        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def normalize2minbound(self):
        self.pointcloud_normalized = copy.deepcopy(self.pointcloud_non_ground_plane)
        points = np.asarray(self.pointcloud_normalized.points)
        self.point = copy.deepcopy(self.pointcloud_non_ground_plane)


        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]

        [x_max, y_max, z_max] = self.pointcloud_normalized.get_min_bound()
        # max_bound = self.pointcloud_filtered.get_max_bound()
        # x_max = np.max(x)
        # y_max = np.max(y)
        # z_max = np.max(z)

        # point_max = np.array([x_max, y_max, z_max]).reshape(1, 3)
        point_max = np.array([0, 0, 0]).reshape(1, 3)
        self.point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_max))

        x = x - x_max
        y = y - y_max
        # z = z - z_max

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        self.pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def pcd_centroid(self, pcd_format):
        if pcd_format == 'non_ground_plane':
            points = np.asarray(self.pointcloud_non_ground_plane.points)
        elif pcd_format == 'normalized':
            points = np.asarray(self.pointcloud_normalized.points)
        [x, y] = points[:, 0], points[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)


        # [x_mean, y_mean, z_max] = self.pointcloud_non_ground_plane.get_min_bound()
        return x_mean, y_mean

    def filter_by_height(self, height=-0.5):
        points = np.asarray(self.pointcloud.points)
        idx = points[:, 2] < height
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))

    def calculate_plane(self, pcd=None, height=-0.5, thresholdA=0.01):
        # find a plane by removing some of the points at a given height
        # this best estimates a ground plane.

        if pcd is None:
            points = np.asarray(self.pointcloud_filtered.points)
        else:
            points = np.asarray(pcd.points)


        idx = points[:, 2] < height
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(points[idx])
        plane_model, inliers = pcd_plane.segment_plane(distance_threshold=thresholdA, ransac_n=3,
                                                       num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane model calculated: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        return plane_model

    def segment_plane(self, plane_model, pcd=None, thresholdB=0.4):
        """
        filter roughly the points that may belong to the plane.
        then estimate the plane with these points.
        find the distance of the points to the plane and classify
        """
        # find a plane by removing some of the points at a given height
        # this best estimates a ground plane.
        if pcd is None:
            points = np.asarray(self.pointcloud_filtered.points)
        else:
            points = np.asarray(pcd.points)
        [a, b, c, d] = plane_model

        # if self.compute_groundplane_once == True:
        #     idx = points[:, 2] < height
        #     pcd_plane = o3d.geometry.PointCloud()
        #     pcd_plane.points = o3d.utility.Vector3dVector(points[idx])
        #     plane_model, inliers = pcd_plane.segment_plane(distance_threshold=thresholdA, ransac_n=3, num_iterations=1000)
        #     [a, b, c, d] = plane_model
        #     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        #     self.plane_model = plane_model
        #     self.compute_groundplane_once = False
        #
        # else:
        #     [a, b, c, d] = self.plane_model
        """
        Ecuacion del plano para un entorno en concreto: '/home/arvc/Escritorio/develop/Rosbags_Juanjo/Entorno_inicial_secuencia1'
        """
        # a=-0.01265
        # b=-0.006
        # d=1.74877
        # c=0.9999

        # points = np.asarray(pcd.points)
        # inliers_final = []
        # for i in range(len(points)):
        #     dist = np.abs(a*points[i, 0] + b*points[i, 1] + c*points[i, 2] + d)/np.sqrt(a*a+b*b+c*c)
        #     if dist < thresholdB:
        #         inliers_final.append(i)

        dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a * a + b * b + c * c)
        condicion = dist < thresholdB
        inliers_final = np.where(condicion==True)
        inliers_final = inliers_final[0]

        # now select the final pointclouds
        plane_cloud = pcd.select_by_index(inliers_final)
        non_plane_cloud = pcd.select_by_index(inliers_final, invert=True)
        return plane_cloud, non_plane_cloud

    def fpfh_similarity(self, other):
        source = self.pcd_fpfh
        target = other.pcd_fpfh
        source = source.data
        target = target.data

        import torch
        pdist = nn.PairwiseDistance(p=2)
        # target0 = target[:, 0]
        # target0 = target0.reshape(33,1)
        [_, longitud] = source.shape
        # resta = np.repeat(target0, longitud, axis=1)
        #
        # result = pdist(target0,source)


        similarity = 0
        return similarity


    def local_registrationA(self, other, initial_transform):
        """
        Use icp to compute transformation using an initial estimate.
        Method A uses all points and a PointToPlane estimation.
        caution, initial_transform is a np array.
        """
        debug = False
        print("Apply point-to-plane ICP")
        print("Using threshold: ", self.icp_threshold)
        # sigma = 0.1
        # loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        # print("Using robust loss:", loss)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud_filtered, self.pointcloud_filtered, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        # print("Transformation is:")
        # print(reg_p2p.transformation)
        if debug:
            other.draw_registration_result(self, reg_p2p.transformation)
        return HomogeneousMatrix(reg_p2p.transformation), reg_p2p.inlier_rmse

    def local_registrationB(self, other, initial_transform):
        """
        Use icp to compute transformation using an initial estimate.
        Method B segments a ground plane and estimates two different transforms:
            - A: using ground planes tz, alfa and beta are estimated. Point to
            - B: using non ground planes (rest of the points) tx, ty and gamma are estimated
        caution, initial_transform is a np array.
        """
        debug = True
        # if debug:
        #     other.draw_registration_result(self, initial_transform)

        # compute a first transform for tz, alfa, gamma, using ground planes
        reg_p2pa = o3d.pipelines.registration.registration_icp(
            other.pointcloud_ground_plane, self.pointcloud_ground_plane, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print(reg_p2pa)
        # if debug:
        #     other.draw_registration_result(self, reg_p2pa.transformation)
        # compute second transformation using the whole pointclouds. CAUTION: failures in ground plane segmentation
        # do affect this transform if computed with some parts of ground
        reg_p2pb = o3d.pipelines.registration.registration_icp(
            other.pointcloud_filtered, self.pointcloud_filtered, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print(reg_p2pb)

        # if debug:
        #     other.draw_registration_result(self, reg_p2pb.transformation)

        t1 = HomogeneousMatrix(reg_p2pa.transformation).t2v(n=3)
        t2 = HomogeneousMatrix(reg_p2pb.transformation).t2v(n=3)
        # build solution using both solutions
        tx = t2[0]
        ty = t2[1]
        tz = t1[2]
        alpha = t1[3]
        beta = t1[4]
        gamma = t2[5]
        T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))

        if debug:
            other.draw_registration_result(self, T.array)
        return T, reg_p2pb.inlier_rmse


    def local_registrationB(self, other, initial_transform):
        """
        Use icp to compute transformation using an initial estimate.
        Method B segments a ground plane and estimates two different transforms:
            - A: using ground planes tz, alfa and beta are estimated. Point to
            - B: using non ground planes (rest of the points) tx, ty and gamma are estimated
        caution, initial_transform is a np array.
        """
        debug = True
        # if debug:
        #     other.draw_registration_result(self, initial_transform)

        # compute a first transform for tz, alfa, gamma, using ground planes
        reg_p2pa = o3d.pipelines.registration.registration_icp(
            other.pointcloud_ground_plane, self.pointcloud_ground_plane, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print(reg_p2pa)
        # if debug:
        #     other.draw_registration_result(self, reg_p2pa.transformation)
        # compute second transformation using the whole pointclouds. CAUTION: failures in ground plane segmentation
        # do affect this transform if computed with some parts of ground
        reg_p2pb = o3d.pipelines.registration.registration_icp(
            other.pointcloud_filtered, self.pointcloud_filtered, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print(reg_p2pb)

        # if debug:
        #     other.draw_registration_result(self, reg_p2pb.transformation)

        t1 = HomogeneousMatrix(reg_p2pa.transformation).t2v(n=3)
        t2 = HomogeneousMatrix(reg_p2pb.transformation).t2v(n=3)
        # build solution using both solutions
        tx = t2[0]
        ty = t2[1]
        tz = t1[2]
        alpha = t1[3]
        beta = t1[4]
        gamma = t2[5]
        T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))

        if debug:
            other.draw_registration_result(self, T.array)
        return T, reg_p2pb.inlier_rmse

    def local_registrationP2P(self, other, initial_transform):


        result = self.point2point_registration(other, initial_transform)

            # other.draw_registration_result(self, np.eye(4))
            # other.draw_registration_result(self, T.array)
        if DEBUGGING_PARAMETERS.plot_initial_tranform:
            other.draw_registration_result(self, initial_transform)
        if DEBUGGING_PARAMETERS.plot_registration_result:
            other.draw_registration_result(self, result.transformation)

        atb = HomogeneousMatrix(result.transformation)

        return atb, result.inlier_rmse

    def global_registrationD(self, other):
        """
        Method based on Scan Context plus correlation.
        Two scan context descriptors are found.
        """
        print('Computing global registration using Scan Context')
        debug = True
        if debug:
            other.draw_registration_result(self, np.eye(4))
            # self.pointcloud.paint_uniform_color([1, 0, 0])
            # other.pointcloud.paint_uniform_color([0, 0, 1])
            # o3d.visualization.draw_geometries([self.pointcloud, other.pointcloud])

        # sample down points
        # using points that do not belong to ground
        voxel_down_sample = 0.1

        pcd1 = self.pointcloud.voxel_down_sample(voxel_size=voxel_down_sample)
        pcd2 = other.pointcloud.voxel_down_sample(voxel_size=voxel_down_sample)
        sc1 = self.scdescriptor.compute_descriptor(pcd1.points)
        sc2 = other.scdescriptor.compute_descriptor(pcd2.points)

        # pcd1 = self.pointcloud_filtered.voxel_down_sample(voxel_size=voxel_down_sample)
        # pcd2 = other.pointcloud_filtered.voxel_down_sample(voxel_size=voxel_down_sample)
        # sc1 = self.scdescriptor.compute_descriptor(pcd1.points)
        # sc2 = other.scdescriptor.compute_descriptor(pcd2.points)

        # pcd1 = self.pointcloud_non_ground_plane.voxel_down_sample(voxel_size=voxel_down_sample)
        # pcd2 = other.pointcloud_non_ground_plane.voxel_down_sample(voxel_size=voxel_down_sample)
        # sc1 = self.scdescriptor.compute_descriptor(pcd1.points)
        # sc2 = other.scdescriptor.compute_descriptor(pcd2.points)

        # pcd1 = self.pointcloud_ground_plane.voxel_down_sample(voxel_size=voxel_down_sample)
        # pcd2 = other.pointcloud_ground_plane.voxel_down_sample(voxel_size=voxel_down_sample)
        # sc1 = self.scdescriptor.compute_descriptor(pcd1.points)
        # sc2 = other.scdescriptor.compute_descriptor(pcd2.points)

        if debug:
            plt.figure()
            # plt.imshow(sc1, cmap='gray')
            plt.imshow(sc1)
            plt.figure()
            # plt.imshow(sc2, cmap='gray')
            plt.imshow(sc2)
            plt.figure()
            # plt.imshow(sc2, cmap='gray')
            plt.imshow(sc1-sc2)

        gamma, prob = self.scdescriptor.maximize_correlation(other.scdescriptor)
        # assuming a rough SE2 transformation here
        T = HomogeneousMatrix(np.array([0, 0, 0]), Euler([0, 0, gamma]))

        if debug:
            other.draw_registration_result(self, T.array)
        return T, prob

    # def centroids_distance(self, other):

    # def global_registration(source_down, target_down, source_fpfh,
    #                                 target_fpfh, voxel_size):
    def initial_registration_fpfh(self, other):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            other.pointcloud_non_ground_plane, self.pointcloud_non_ground_plane, other.pcd_fpfh, self.pcd_fpfh, True,
            self.fpfh_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.fpfh_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        return result

    def point2point_registration(self, other, initial_transform):
        result = o3d.pipelines.registration.registration_icp(other.pointcloud_non_ground_plane, self.pointcloud_non_ground_plane, 2, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return result

    def global_registrationFPFH(self, other):
        result_fpfh = self.initial_registration_fpfh(other)


        result_plane = o3d.pipelines.registration.registration_icp(
            other.pointcloud_ground_plane, self.pointcloud_ground_plane, self.icp_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        t1 = HomogeneousMatrix(result_plane.transformation).t2v(n=3)
        t2 = HomogeneousMatrix(result_fpfh.transformation).t2v(n=3)
        # build solution using both solutions
        tx = t2[0]
        ty = t2[1]
        tz = t1[2]
        alpha = t1[3]
        beta = t1[4]
        gamma = t2[5]
        T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))
        result = self.point2point_registration(other, T.array)

        if DEBUGGING_PARAMETERS.plot_registration_result:
            other.draw_registration_result(self, T.array)
            other.draw_registration_result(self, result.transformation)

        atb = HomogeneousMatrix(result.transformation)

        return atb, result.inlier_rmse

    def draw_registration_result(self, other, transformation):
        # source_temp = copy.deepcopy(self.pointcloud)
        # target_temp = copy.deepcopy(other.pointcloud)
        source_temp = copy.deepcopy(self.pointcloud_non_ground_plane)
        target_temp = copy.deepcopy(other.pointcloud_non_ground_plane)
        # source_temp = copy.deepcopy(self.pointcloud_filtered)
        # target_temp = copy.deepcopy(other.pointcloud_filtered)
        # source_temp.paint_uniform_color([1, 0, 0])
        # target_temp.paint_uniform_color([0, 0, 1])
        source_temp.paint_uniform_color([0.0, 0.7, 0.8])
        target_temp.paint_uniform_color([0.5, 0.5, 0.5])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def draw_pointclouds(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud_normalized)
        target_temp = copy.deepcopy(other.pointcloud_normalized)
        # source_temp = copy.deepcopy(self.pointcloud_filtered)
        # target_temp = copy.deepcopy(other.pointcloud_filtered)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)

        source_point = copy.deepcopy(self.point)
        target_point = copy.deepcopy(other.point)
        source_point.paint_uniform_color([0, 0, 0])
        target_point.paint_uniform_color([0, 0, 0])
        source_point.transform(transformation)

        # source_points_bbox = copy.deepcopy(self.points_bbox)
        # target_points_bbox = copy.deepcopy(other.points_bbox)
        # source_points_bbox.paint_uniform_color([1, 1, 0])
        # target_points_bbox.paint_uniform_color([0, 1, 0])
        # source_points_bbox.transform(transformation)


        # o3d.visualization.draw_geometries([source_temp, target_temp, source_point, target_point, source_points_bbox, target_points_bbox])
        o3d.visualization.draw_geometries([source_temp, target_temp, source_point, target_point])

    def draw_pointclouds_bbox(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud_normalized)
        target_temp = copy.deepcopy(other.pointcloud_normalized)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)

        source_point = copy.deepcopy(self.point)
        target_point = copy.deepcopy(other.point)
        source_point.paint_uniform_color([0, 0, 0])
        target_point.paint_uniform_color([0, 0, 0])
        source_point.transform(transformation)

        source_points_bbox = copy.deepcopy(self.points_bbox)
        target_points_bbox = copy.deepcopy(other.points_bbox)
        source_points_bbox.paint_uniform_color([1, 1, 0])
        target_points_bbox.paint_uniform_color([0, 1, 0])
        source_points_bbox.transform(transformation)

        o3d.visualization.draw_geometries([source_temp, target_temp, source_point, target_point, source_points_bbox, target_points_bbox])
        # o3d.visualization.draw_geometries([source_temp, target_temp, source_point, target_point])

    def draw_pointcloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])

    def numpy2pointcloud(self, xyz):
        self.pointcloud.points = o3d.utility.Vector3dVector(xyz)

    def set_global_transform(self, transform):
        self.transform = transform
        return

    def transform_to_global(self, point_cloud_sampling=10):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        if self.transform is None:
            return None
        T = HomogeneousMatrix(self.transform)
        pointcloud = self.pointcloud.uniform_down_sample(every_k_points=point_cloud_sampling)
        return pointcloud.transform(T.array)

    def transform_by_T(self, T):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        return self.pointcloud.transform(T.array)

    def kd_tree(self, other, transformation):

        # pcd.paint_uniform_color([0.5, 0.5, 0.5])

        source_temp = copy.deepcopy(self.pointcloud_non_ground_plane)
        target_temp = copy.deepcopy(other.pointcloud_non_ground_plane)
        # source_temp.paint_uniform_color([0.5, 0.5, 0.5])
        target_temp.paint_uniform_color([0.0, 0.7, 0.8])

        target_temp.transform(transformation.array)

        punto = 1500
        source_temp.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_tree = o3d.geometry.KDTreeFlann(target_temp)
        print("Paint the 1500th point red.")
        source_temp.colors[punto] = [1, 0, 0]

        # print("Find its 500 nearest neighbors, and paint them blue.")
        # [k, idx, _] = pcd_tree.search_knn_vector_3d(source_temp.points[1500], 1000)
        # np.asarray(target_temp.colors)[idx[1:], :] = [0, 0, 1]

        print("Find its neighbors with distance less than 0.2, and paint them green.")
        radio = 1
        [k, idx, _] = pcd_tree.search_radius_vector_3d(source_temp.points[punto], radio)

        np.asarray(target_temp.colors)[idx[1:], :] = [0, 1, 0]
        o3d.visualization.draw_geometries([target_temp, source_temp])

        return k

    def overlap_3d(self, other, transformation):

        debug = True
        source_temp = copy.deepcopy(self.pointcloud_non_ground_plane)
        target_temp = copy.deepcopy(other.pointcloud_non_ground_plane)
        if debug:
            source_temp.paint_uniform_color([0.5, 0.5, 0.5])
            target_temp.paint_uniform_color([0.0, 0.7, 0.8])

        target_temp.transform(transformation.array)
        pcd_tree = o3d.geometry.KDTreeFlann(target_temp)
        radio = 0.2
        print("Find its neighbors with distance less than 0.2, and paint them green.")
        matches = 0

        for punto in range(0, len(source_temp.points)):
            if punto == 0:
                [k, indices, _] = pcd_tree.search_radius_vector_3d(source_temp.points[punto], radio)
            else:
                [k, idx, _] = pcd_tree.search_radius_vector_3d(source_temp.points[punto], radio)
                indices = np.concatenate((indices, idx), axis=0)
            if k > 0:
                matches += 1

        overlap = matches / len(source_temp.points)
        print(overlap)
        if debug:
            indices = np.unique(indices)
            np.asarray(target_temp.colors)[indices [1:], :] = [1, 0, 0]
            o3d.visualization.draw_geometries([source_temp, target_temp])
        return overlap

    def overlap(self, source_temp, target_temp, transformation):
        debug = False
        target_temp.transform(transformation)
        pcd_tree = o3d.geometry.KDTreeFlann(target_temp)
        radio = EXP_PARAMETERS.overlap_radius
        # print("Find its neighbors with distance less than 0.2, and paint them green.")
        matches = 0

        for punto in range(0, len(source_temp.points)):
            if punto == 0:
                [k, indices, _] = pcd_tree.search_radius_vector_3d(source_temp.points[punto], radio)
            else:
                [k, idx, _] = pcd_tree.search_radius_vector_3d(source_temp.points[punto], radio)
                indices = np.concatenate((indices, idx), axis=0)
            if k > 0:
                matches += 1
        indices = np.unique(indices)

        if debug:
            np.asarray(target_temp.colors)[indices[1:], :] = [1, 0, 0]
            # np.asarray(source_temp.colors)[source_indices[1:], :] = [1, 0, 0]
            o3d.visualization.draw_geometries([source_temp, target_temp])
        return matches, indices

    def pairwise_overlap(self, other, transformation):
        source_temp0 = copy.deepcopy(self.pointcloud_non_ground_plane)
        target_temp0 = copy.deepcopy(other.pointcloud_non_ground_plane)
        source_temp1 = copy.deepcopy(self.pointcloud_non_ground_plane)
        target_temp1 = copy.deepcopy(other.pointcloud_non_ground_plane)

        # source_temp0 = copy.deepcopy(self.pointcloud_training)
        # target_temp0 = copy.deepcopy(other.pointcloud_training)
        # source_temp1 = copy.deepcopy(self.pointcloud_training)
        # target_temp1 = copy.deepcopy(other.pointcloud_training)

        if DEBUGGING_PARAMETERS.plot_scan_overlap:
            source_temp0.paint_uniform_color([0.5, 0.5, 0.5])
            target_temp0.paint_uniform_color([0.0, 0.7, 0.8])
            source_temp1.paint_uniform_color([0.5, 0.5, 0.5])
            target_temp1.paint_uniform_color([0.0, 0.7, 0.8])

        target_matches, target_indices = self.overlap(source_temp0, target_temp0, transformation.array)
        source_matches, source_indices = self.overlap(target_temp1, source_temp1, np.linalg.inv(transformation.array))

        overlap = (source_matches + target_matches) / (len(source_temp0.points) + len(target_temp0.points))
        # print('Pairwise overlap:', overlap)
        # print('Source overlap:', source_matches/len(source_temp0.points))
        # print('Target overlap:', target_matches / len(target_temp0.points))
        if DEBUGGING_PARAMETERS.plot_scan_overlap:
            np.asarray(target_temp0.colors)[target_indices[1:], :] = [1, 0, 0] #points matched in red
            np.asarray(source_temp0.colors)[source_indices[1:], :] = [1, 0, 0] #points matched in red
            o3d.visualization.draw_geometries([source_temp0, target_temp0])
        return overlap













