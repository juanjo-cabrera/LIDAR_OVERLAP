import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from config import TRAINING_PARAMETERS, EXP_PARAMETERS, ICP_PARAMETERS
import pandas as pd
import numpy as np
from scan_tools.keyframe import KeyFrame
import MinkowskiEngine as ME
# from examples.classification_modelnet40 import *
# from scripts.examples.classification_modelnet40 import *
# from ml_tools.FCNN import MinkowskiFCNN
from eurocreader.eurocreader_outdoors import EurocReader
from google_maps_plotter.custom_plotter import *
from sklearn.neighbors import KDTree
from tqdm import tqdm
from time import sleep
from kittireader.kittireader import KittiReader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from ml_tools.minkunet import MinkUNet34C
from ml_tools.VGG16 import *
from ml_tools.VGG11 import *
from ml_tools.VGG13 import *
from ml_tools.VGG19 import *
from ml_tools.pointnet import *
from MinkLoc3Dv2_models.model_factory import MinkLoc3Dv2
# from torchvision.models import VGG16_Weights, vgg16
from PIL import Image

vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

class TrainingDataset(Dataset):
    def __init__(self, dir=None, transform=None):
        if dir is None:
            self.root_dir = TRAINING_PARAMETERS.training_path
        else:
            self.root_dir = dir
        labels_dir = self.root_dir + '/anchor_1m_uniform_v2.csv'
        if self.root_dir.find('Kitti') == -1:
            self.scans_dir = self.root_dir + '/robot0/lidar/data/'
        else:
            self.scans_dir = self.root_dir + '/velodyne/'

        df = pd.read_csv(labels_dir)
        self.reference_timestamps = np.array(df["Reference timestamp"])
        self.other_timestamps = np.array(df["Other timestamp"])
        self.overlap = np.array(df["Overlap"])
        self.transform = transform

    def __getitem__(self, idx):
        reference_timestamp = self.reference_timestamps[idx]
        reference_kf = KeyFrame(directory=self.root_dir, scan_time=reference_timestamp)
        reference_kf.load_pointcloud()
        reference_range, reference_detected_points = get_depth_image(reference_kf)

        reference_img = Image.fromarray(reference_range)
        reference_img = reference_img.convert("RGB")
        # reference_range = reference_range.reshape(1, 64, 900)
        # plot_range_image(reference_range)
        # plot_range_image(reference_detected_points)

        other_timestamp = self.other_timestamps[idx]
        other_kf = KeyFrame(directory=self.root_dir, scan_time=other_timestamp)
        other_kf.load_pointcloud()
        other_range, other_detected_points = get_depth_image(other_kf)
        other_img = Image.fromarray(other_range)
        other_img = other_img.convert("RGB")
        # other_range = other_range.reshape(1, 64, 900)
        diferencia = 1 - np.array([self.overlap[idx]])
        if self.transform:
            reference_img = self.transform(reference_img)
            other_img = self.transform(other_img)
        return reference_img, other_img, diferencia

    def __len__(self):
        return len(self.overlap)


class GroundTruthDataset(Dataset):
    def __init__(self, data, transform=None):
        self.root_dir = TRAINING_PARAMETERS.groundtruth_path
        self.scan_times, _, self.pos = data
        self.transform = transform
        # kf = KeyFrame(directory=self.root_dir, scan_time=self.scan_times[0])
        # kf.load_pointcloud()
        # pointcloud_filtered = kf.filter_by_radius(0, TRAINING_PARAMETERS.max_radius)
        # self.plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        range, detected_points = get_depth_image(kf)
        img = Image.fromarray(range)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, position

    def __len__(self):
        return len(self.scan_times)


class ValidationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.root_dir = TRAINING_PARAMETERS.validation_path
        self.transform = transform
        # Prepare data
        # euroc_read = EurocReader(directory=self.root_dir)
        # self.scan_times, _, self.pos = euroc_read.prepare_gps_data(deltaxy=5)
        self.scan_times, _, self.pos = data
        # kf = KeyFrame(directory=self.root_dir, scan_time=self.scan_times[0])
        # kf.load_pointcloud()
        # pointcloud_filtered = kf.filter_by_radius(0, TRAINING_PARAMETERS.max_radius)
        # self.plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        range, detected_points = get_depth_image(kf)
        img = Image.fromarray(range)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, position

    def __len__(self):
        return len(self.scan_times)



class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive




class SiameseNetwork(nn.Module):
    #DEFINIMOS LA ESTRUCTURA DE LA RED NEURONAL
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.conv1 = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(3, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True))
        # self.conv2 = nn.Sequential(
        #     # conv2
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True))
        # self.conv3 = nn.Sequential(
        #     # conv3
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True))
        # self.conv4 = nn.Sequential(
        #     # conv4
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True))
        # self.conv5 = nn.Sequential(
        #     # conv5
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True)
        # )
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))
    #Aqui definimos como estan conectadas las capas entre sí, como pasamos el input al output, para cada red
    def forward(self, x):#toma con la variable x el input
        verbose = False

        if verbose:
            print("Input: ", x.size())

        # output = self.conv1(x)
        # output = self.conv2(output)
        # output = self.conv3(output)
        # output = self.conv4(output)
        # output = self.conv5(output)
        output = self.features(x)
        if verbose:
            print("Output matricial: ", output.size())

        output = self.avgpool(output)
        if verbose:
            print("Output avgpool: ", output.size())
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return output


def range_projection_kitti(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
          current_vertex: raw point clouds
        Returns:
          proj_range: projected range image with depth, each pixel contains the corresponding depth
          proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
          proj_intensity: each pixel contains the corresponding intensity
          proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx

def spherical_projection_ouster(homogeneous_points, fov=45, proj_W=512, proj_H=128, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
        Returns:
           proj_range: projected range image with depth, each pixel contains the corresponding depth
           proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
           proj_intensity: each pixel contains the corresponding intensity
           proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
     """
    fov = np.pi * fov / 180.0  # pasamos a radianes
    depth = np.linalg.norm(homogeneous_points[:, :3], 2, axis=1)
    homogeneous_points = homogeneous_points[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = homogeneous_points[:, 0]
    scan_y = homogeneous_points[:, 1]
    scan_z = homogeneous_points[:, 2]
    intensity = homogeneous_points[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    # proj_y = 0.5 * (pitch / (fov/2) + 1.0)  # in [0.0, 1.0] MI INTERPRETACIÓN
    proj_y = 1.0 - (pitch + abs(fov / 2)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # tambien np.round
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx

def get_depth_image(kf):
    current_homogeneous_points = kf.points2homogeneous(pre_process=False)
    current_range, project_points, _, _ = range_projection_kitti(current_homogeneous_points)
    current_detected_points = project_points[
        current_range > 0]  # filtra los puntos que dan en el infinito y devuelven -1
    return current_range, current_detected_points



def get_latent_vectors(dataloader, model, main_device):
    torch.cuda.set_device(main_device)
    model.to(main_device)
    model.eval()
    all_descriptors = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            pcd, poses = data

            # input = ME.TensorField(
            #     features=feat.to(dtype=torch.float32),
            #     coordinates=pcd.to(dtype=torch.float32),
            #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            #     device=main_device,
            # )
            batched_descriptor = model(pcd.to(main_device))
            batched_descriptor = batched_descriptor.detach().cpu()
            if i == 0:
                all_descriptors = batched_descriptor
                continue

            all_descriptors = torch.cat((all_descriptors, batched_descriptor), dim=0)

    return all_descriptors.numpy()


def get_recall(map_output, queries_output, all_true_neighbors):
    # Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(map_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(map_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        true_neighbors = all_true_neighbors[i]
        if len(true_neighbors[0]) == 0:
            continue    # Get recall only if map and validation trajectories are the same
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])): # recorremos cada uno de los 25 vecinos más cercanos
            if indices[0][j] in true_neighbors[0]: # si el indice del vecino predicho esta entre los reales:
                if j == 0: # Si acertamos en el vecino más cercano
                    similarity = np.dot(queries_output[i], map_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors[0])))) > 0:
            # de indices coges los n elementos más cercanos, donde n viende dado el threshold
            # si estos indices seleccionados coinciden con algun vecino verdadero, entonces entra
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall


def get_position_error(queries_descriptors, map_descriptors, queries_poses, map_poses):
    k = 0
    errors = []
    map_feat_tree = KDTree(map_descriptors)
    num_neighbors = 1
    for query_descriptor in queries_descriptors:
        desc_distances, indices = map_feat_tree.query(np.array([query_descriptor]), k=num_neighbors)
        predicted_pose = map_poses[indices]
        real_pose = queries_poses[k]
        pose_error = np.linalg.norm(predicted_pose - real_pose)
        errors.append(pose_error)
        k += 1
    errors = np.array(errors)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    return mean_error, median_error

def compute_validation(model, query_dataloader, map_dataloader, true_neighbors, queries_poses, map_poses):
    device1 = torch.device("cuda:1")

    queries_descriptors = get_latent_vectors(dataloader=query_dataloader, model=model,
                                                                  main_device=device1)
    map_descriptors = get_latent_vectors(dataloader=map_dataloader, model=model,
                                                                  main_device=device1)

    mean_error, median_error = get_position_error(queries_descriptors, map_descriptors, queries_poses, map_poses)
    print('\n\nValidation results \n Mean pose error: {} meters, Median error: {} meters'.format(mean_error, median_error))

    recall, top1_similarity_score, one_percent_recall = get_recall(map_descriptors, queries_descriptors, true_neighbors)
    average_similarity = np.mean(top1_similarity_score)
    print(' Avg.top 1 % recall: {} %, Avg.similarity: {}, Avg.recall @ N: {} \n'.format(one_percent_recall, average_similarity, recall))
    return recall[0], mean_error


def reader_manager():
    directory = TRAINING_PARAMETERS.validation_path
    if directory.find('Kitti') == -1:
        val_data, map_data, all_true_neighbors = read_innova_dataset()

    else:
        val_data, map_data, all_true_neighbors = read_kitti_dataset()

    return val_data, map_data, all_true_neighbors

def read_innova_dataset():
    # Prepare data
    euroc_read_validation = EurocReader(directory=TRAINING_PARAMETERS.validation_path)
    scan_times_val, lat_lon_val, utm_val = euroc_read_validation.prepare_gps_data(deltaxy=5)
    val_data = [scan_times_val, lat_lon_val, utm_val]

    euroc_read_groundtruth = EurocReader(directory=TRAINING_PARAMETERS.groundtruth_path)
    scan_times_map, lat_lon_map, utm_map = euroc_read_groundtruth.prepare_gps_data(deltaxy=EXP_PARAMETERS.exp_deltaxy)
    map_data = [scan_times_map, lat_lon_map, utm_map]

    map_poses_tree = KDTree(utm_map)
    all_true_neighbors = []
    for query_pose in utm_val:
        indexes = map_poses_tree.query_radius(np.array([query_pose]), r=TRAINING_PARAMETERS.success_radius)
        all_true_neighbors.append(indexes)
    return val_data, map_data, all_true_neighbors

def read_kitti_dataset():
    # Prepare data
    kitti_read = KittiReader(directory=TRAINING_PARAMETERS.validation_path)
    scan_times_map, scan_times_val, utm_map, utm_val = kitti_read.prepare_kitti_evaluation(deltaxy_map=5, deltaxy_val=30)

    lat_lon_val = -1
    lat_lon_map = -1
    val_data = [scan_times_val, lat_lon_val, utm_val]
    map_data = [scan_times_map, lat_lon_map, utm_map]

    map_poses_tree = KDTree(utm_map)
    all_true_neighbors = []
    for query_pose in utm_val:
        indexes = map_poses_tree.query_radius(np.array([query_pose]), r=TRAINING_PARAMETERS.success_radius)
        all_true_neighbors.append(indexes)
    return val_data, map_data, all_true_neighbors

def load_validation_data():
    # Prepare data
    euroc_read_validation = EurocReader(directory=TRAINING_PARAMETERS.validation_path)
    scan_times_val, lat_lon_val, utm_val = euroc_read_validation.prepare_gps_data(deltaxy=5)
    val_data = [scan_times_val, lat_lon_val, utm_val]

    euroc_read_groundtruth = EurocReader(directory=TRAINING_PARAMETERS.groundtruth_path)
    scan_times_map, lat_lon_map, utm_map = euroc_read_groundtruth.prepare_gps_data(deltaxy=EXP_PARAMETERS.exp_deltaxy)
    map_data = [scan_times_map, lat_lon_map, utm_map]

    map_poses_tree = KDTree(utm_map)
    all_true_neighbors = []
    for query_pose in utm_val:
        indexes = map_poses_tree.query_radius(np.array([query_pose]), r=TRAINING_PARAMETERS.success_radius)
        all_true_neighbors.append(indexes)
    return val_data, map_data, all_true_neighbors


def vis_poses(validation, map):
    """Visualize the trajectory"""
    # set up plot
    fig, ax = plt.subplots()
    # map poses
    ax.scatter(validation[:, 1]*-1, validation[:, 0], c='red', s=10)
    ax.scatter(map[:, 1]*-1, map[:, 0], c='blue', s=10)

    ax.axis('square')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Poses')
    ax.legend(['Validation', 'Map'])
    plt.show()


def visualize_trajectories(val_data, map_data):
    # Prepare data
    scan_times_val, lat_lon_val, utm_val = val_data
    scan_times_map, lat_lon_map, utm_map = map_data

    if lat_lon_map == -1:
        vis_poses(utm_val, utm_map)
    else:
        gmap = CustomGoogleMapPlotter(EXP_PARAMETERS.origin_lat, EXP_PARAMETERS.origin_lon, zoom=20,
                                      map_type='satellite')

        gmap.pos_scatter(lat_lon_val[:, 0], lat_lon_val[:, 1], color='orange')
        gmap.pos_scatter(lat_lon_map[:, 0], lat_lon_map[:, 1], color='blue')
        gmap.draw(TRAINING_PARAMETERS.validation_path + '/map2.html')

def load_training_sets():
    sequences = [3, 4, 5, 6, 7, 8, 9, 10]
    base_dir = '/home/arvc/Juanjo/Datasets/KittiDataset/sequences/0'
    for i in range(0, len(sequences)):
        if sequences[i] == 10:
            dir = '/home/arvc/Juanjo/Datasets/KittiDataset/sequences/10'
        else:
            dir = base_dir + str(sequences[i])
        if i == 0:
            train_sets = TrainingDataset(dir=dir)
            print(dir, ' ------> ', len(train_sets), ' combinations')
        else:
            train_set = TrainingDataset(dir=dir)
            print(dir, ' ------> ', len(train_set), ' combinations')
            train_sets = torch.utils.data.ConcatDataset([train_sets, train_set])
    print('IN TOTAL ------> ', len(train_sets), ' combinations')
    return train_sets

STR2NETWORK = dict(
    # pointnet=PointNet,
    # minkpointnet=MinkowskiPointNet,
    # minkfcnn=MinkowskiFCNN,
    # minksplatfcnn=MinkowskiSplatFCNN,
    VGG16=VGG16,
    MinkUNet=MinkUNet34C,
    VGG16_avg1024=VGG16_3DNetwork_mod,
    VGG16_cat512=VGG16_3DNetwork_cat,
    VGG16_ext1024=VGG16_ext1024,
    VGG16_reduced256=VGG16_reduced256,
    VGG11=VGG11,
    VGG13=VGG13,
    VGG19=VGG19,
    VGG16_VLAD=VGG16_VLAD

)

def plot_range_image(range):

    # showing image
    plt.imshow(range, cmap='gray')
    plt.axis('off')
    plt.title("Current")
    plt.show()


def main(descriptor_size):
    # val_data, map_data, true_neighbors = load_validation_data()
    val_data, map_data, true_neighbors = reader_manager()
    visualize = False
    if visualize == True:
        visualize_trajectories(val_data, map_data)
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device is: ", device0)
    train_dataset = TrainingDataset(transform=transforms.ToTensor())
    # train_dataset = TrainingDataset_NOoverlap()
    # train_dataset = load_training_sets()
    groundtruth_dataset = GroundTruthDataset(data=map_data, transform=transforms.ToTensor())
    validation_dataset = ValidationDataset(data=val_data, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_PARAMETERS.training_batch_size, shuffle=True)
    groundtruth_dataloader = DataLoader(groundtruth_dataset, batch_size=TRAINING_PARAMETERS.groundtruth_batch_size, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=TRAINING_PARAMETERS.validation_batch_size, shuffle=False)

    # initialize model
    # net = STR2NETWORK['VGG16'](
    #     in_channel=3, out_channel=TRAINING_PARAMETERS.output_size, D=3).to(device0)
    # net_arquitecture = 'MinkUNet'
    net_arquitecture = 'VGG16_2d'
    # net = STR2NETWORK[net_arquitecture](
    #     in_channels=3, out_channels=descriptor_size, D=3).to(device0)

    net = SiameseNetwork().to(device0)
    # net_arquitecture = 'MinkowskiPointNet'
    # net = MinkowskiPointNet(
    #     in_channel=3, out_channel=20, embedding_channel=1024, dimension=3
    # ).to(device0)
    # net = MinkLoc3Dv2.to(device0)
    # net_arquitecture = 'MinkLoc3Dv2'
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")

    criterion = ContrastiveLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())

    # train model
    counter = []
    error_history = []
    recall_at1_history = []

    last_errors = []
    error_history.append(1000)
    recall_at1_history.append(0)
    # net_name = net_arquitecture + 'maxpool_512_' + str(descriptor_size) + '_04_1m_recall'
    net_name = net_arquitecture + '_04_1m_recall'
    net.train()

    for epoch in range(TRAINING_PARAMETERS.number_of_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            i = 0
            for training_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                # Get new data
                ref_pcd, other_pcd, label = training_data
                ref_pcd, other_pcd, label = ref_pcd.to(device0), other_pcd.to(device0), label.to(device0)

                # Forward

                ref_desc = net(ref_pcd)
                other_desc = net(other_pcd)
                loss = criterion(ref_desc, other_desc, label)  # For tensor field
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    recall_at1, mean_error = compute_validation(model=net, query_dataloader=validation_dataloader,
                                       map_dataloader=groundtruth_dataloader, true_neighbors=true_neighbors,
                                       queries_poses=val_data[2], map_poses=map_data[2])

                    min_error = np.min(error_history)
                    max_recall = np.max(recall_at1_history)
                    recall_at1_history.append(recall_at1)
                    error_history.append(mean_error)
                    # if mean_error < min_error:
                    if recall_at1>max_recall:
                        # save model
                        # torch.save(net.state_dict(), net_name + str(mean_error))
                        torch.save(net.state_dict(), net_name + str(recall_at1) + '_epoch' + str(epoch) + '_iter' + str(i))
                    # Model to training device
                    net.to(device0)
                    torch.cuda.set_device(device0)
                    net.train(mode=True)

                    if TRAINING_PARAMETERS.complete_epochs == False:

                        if len(last_errors) < 10:
                            last_errors.append(mean_error)
                            counter.append(i)
                        elif len(last_errors) == 10:
                            del last_errors[0]
                            del counter[0]
                            last_errors.append(mean_error)
                            counter.append(i)

                        if len(last_errors) > 2:
                            error_tendency = np.polyfit(x=np.array(last_errors), y=np.array(counter), deg=1)
                            error_tendency = error_tendency[0]
                            if error_tendency < 0:
                                print('\nAprendiendo\n')
                            else:
                                break
                                break

                i += 1


                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)


if __name__ == '__main__':
    descriptors_sizes = [512]
    for descriptors_size in descriptors_sizes:
        main(descriptors_size)