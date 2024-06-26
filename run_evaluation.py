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




class GroundTruthDataset(Dataset):
    def __init__(self, data, transform=None):
        self.root_dir = TRAINING_PARAMETERS.groundtruth_path
        # Prepare data
        # euroc_read = EurocReader(directory=self.root_dir)
        # self.scan_times, self.pos, _, _ = euroc_read.prepare_experimental_data(
        #     deltaxy=EXP_PARAMETERS.exp_deltaxy,
        #     deltath=EXP_PARAMETERS.exp_deltath,
        #     nmax_scans=EXP_PARAMETERS.exp_long,
        #     gps_mode='utm')
        self.scan_times, _, self.pos = data
        kf = KeyFrame(directory=self.root_dir, scan_time=self.scan_times[0])
        kf.load_pointcloud()
        pointcloud_filtered = kf.filter_by_radius(0, TRAINING_PARAMETERS.max_radius)
        self.plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        pcd, features = kf.training_preprocess(plane_model=self.plane_model)

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return pcd, features, position

    def __len__(self):
        return len(self.scan_times)


class ValidationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.root_dir = TRAINING_PARAMETERS.validation_path
        # Prepare data
        # euroc_read = EurocReader(directory=self.root_dir)
        # self.scan_times, _, self.pos = euroc_read.prepare_gps_data(deltaxy=5)
        self.scan_times, _, self.pos = data
        kf = KeyFrame(directory=self.root_dir, scan_time=self.scan_times[0])
        kf.load_pointcloud()
        pointcloud_filtered = kf.filter_by_radius(0, TRAINING_PARAMETERS.max_radius)
        self.plane_model = kf.calculate_plane(pcd=pointcloud_filtered)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        pcd, features = kf.training_preprocess(plane_model=self.plane_model)

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return pcd, features, position

    def __len__(self):
        return len(self.scan_times)


class ReferenceDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.directory
        labels_dir = TRAINING_PARAMETERS.directory + '/labelling.csv'
        self.scans_dir = TRAINING_PARAMETERS.directory + '/robot0/lidar/data/'

        df = pd.read_csv(labels_dir)
        self.reference_timestamps = np.array(df["Reference timestamp"])
        self.overlap = np.array(df["Overlap"])
        self.transform = transform

    def __getitem__(self, idx):
        reference_timestamp = self.reference_timestamps[idx]
        reference_kf = KeyFrame(directory=self.root_dir, scan_time=reference_timestamp)
        reference_kf.load_pointcloud()
        reference_pcd, reference_features = reference_kf.training_preprocess()

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return reference_pcd, reference_features, np.array([self.overlap[idx]])

    def __len__(self):
        return len(self.overlap)

class OtherDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.directory
        labels_dir = TRAINING_PARAMETERS.directory + '/labelling.csv'
        self.scans_dir = TRAINING_PARAMETERS.directory + '/robot0/lidar/data/'

        df = pd.read_csv(labels_dir)
        self.other_timestamps = np.array(df["Other timestamp"])
        self.overlap = np.array(df["Overlap"])
        self.transform = transform

    def __getitem__(self, idx):
        other_timestamp = self.other_timestamps[idx]
        other_kf = KeyFrame(directory=self.root_dir, scan_time=other_timestamp)
        other_kf.load_pointcloud()
        other_pcd, other_features = other_kf.training_preprocess()

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return other_pcd, other_features

    def __len__(self):
        return len(self.overlap)

class ValidationExample():
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.validation_path
        # Prepare data
        euroc_read = EurocReader(directory=self.root_dir)
        self.scan_times, self.pos, _, _ = euroc_read.prepare_ekf_data(
            deltaxy=EXP_PARAMETERS.exp_deltaxy,
            deltath=EXP_PARAMETERS.exp_deltath,
            nmax_scans=EXP_PARAMETERS.exp_long)

        self.transform = transform

    def get_random_example(self):
        random_number = random.choice(np.arange(0, len(self.scan_times)))
        random_scan = self.scan_times[random_number]
        random_pos = self.pos[random_number]


        kf = KeyFrame(directory=self.root_dir, scan_time=random_scan)
        kf.load_pointcloud()
        random_pcd, random_features = kf.training_preprocess()
        N = len(random_pcd)
        batched_pcd = torch.zeros((N, 4), dtype=torch.int32, device=device0)
        batched_pcd[:, 1:] = torch.from_numpy(random_pcd).to(dtype=torch.float32)

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return batched_pcd, torch.from_numpy(random_features).to(dtype=torch.float32), random_pos


class VGG16_3DNetwork(nn.Module):
    def __init__(self, in_channel, out_channel, D):
        super(VGG16_3DNetwork, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channel,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
            # ME.MinkowskiMaxPooling(kernel_size=3, stride=1, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
            # ME.MinkowskiMaxPooling(kernel_size=3, stride=1, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(256), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(256), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(256), ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiConvolution(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiGlobalPooling(),
            # ME.MinkowskiLinear(512, 512),
            # ME.MinkowskiLinear(512, 128),
            ME.MinkowskiLinear(512, out_channel))

    def forward(self, x):
        x = x.sparse()
        embedding = self.net(x).F
        if TRAINING_PARAMETERS.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings
        return embedding


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def training_collation_fn(data_labels):
    reference_pcd, reference_features, other_pcd, other_features,  labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    reference_pcd_batch = ME.utils.batched_coordinates(reference_pcd)
    other_pcd_batch = ME.utils.batched_coordinates(other_pcd)

    # Concatenate all lists
    ref_feats_batch = torch.from_numpy(np.concatenate(reference_features, 0)).float()
    other_feats_batch = torch.from_numpy(np.concatenate(other_features, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()

    return reference_pcd_batch, ref_feats_batch, other_pcd_batch, other_feats_batch, labels_batch

def ground_collation_fn(data_labels):
    pcd, features, position = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    pcd_batch = ME.utils.batched_coordinates(pcd)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(features, 0)).float()
    position_batch = torch.from_numpy(np.concatenate(position, 0)).float()

    return pcd_batch, feats_batch, position_batch

def get_latent_vectors(dataloader, model, main_device):
    torch.cuda.set_device(main_device)
    model.to(main_device)
    model.eval()
    all_descriptors = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            pcd, feat, poses = data

            input = ME.TensorField(
                features=feat.to(dtype=torch.float32),
                coordinates=pcd.to(dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=main_device,
            )
            batched_descriptor = model(input)
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
    scan_times_map, scan_times_val, utm_map, utm_val = kitti_read.prepare_kitti_evaluation()

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



STR2NETWORK = dict(
    # pointnet=PointNet,
    # minkpointnet=MinkowskiPointNet,
    # minkfcnn=MinkowskiFCNN,
    # minksplatfcnn=MinkowskiSplatFCNN,
    VGG16=VGG16_3DNetwork
)


if __name__ == '__main__':
    val_data, map_data, true_neighbors = reader_manager()
    visualize_trajectories(val_data, map_data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device is: ", device)


    groundtruth_dataset = GroundTruthDataset(data=map_data)
    validation_dataset = ValidationDataset(data=val_data)

    groundtruth_dataloader = DataLoader(groundtruth_dataset, batch_size=TRAINING_PARAMETERS.groundtruth_batch_size, shuffle=False,
                                  collate_fn=ground_collation_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=TRAINING_PARAMETERS.validation_batch_size, shuffle=False,
                                  collate_fn=ground_collation_fn)

    # initialize model
    net = STR2NETWORK['VGG16'](
        in_channel=3, out_channel=TRAINING_PARAMETERS.output_size, D=3).to(device)

    print("===================Network===================")
    print(net)
    print("=============================================\n\n")

    net_name = '/home/arvc/Juanjo/develop/LIDAR_OVERLAP/VGG16_08_1m_recall71.43_epoch0_iter950'
    net.load_state_dict(torch.load(net_name))
    net.eval()


    recall_at1, mean_error = compute_validation(model=net, query_dataloader=validation_dataloader,
                       map_dataloader=groundtruth_dataloader, true_neighbors=true_neighbors,
                       queries_poses=val_data[2], map_poses=map_data[2])



