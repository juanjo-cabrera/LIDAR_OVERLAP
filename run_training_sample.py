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
from scripts.examples.classification_modelnet40 import *
# from ml_tools.FCNN import MinkowskiFCNN
from eurocreader.eurocreader_outdoors import EurocReader
from google_maps_plotter.custom_plotter import *
from kittireader.kittireader import KittiReader

class TrainingDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.training_path
        labels_dir = self.root_dir + '/labelling.csv'
        self.scans_dir = self.root_dir + '/robot0/lidar/data/'

        df = pd.read_csv(labels_dir)
        self.reference_timestamps = np.array(df["Reference timestamp"])
        self.other_timestamps = np.array(df["Other timestamp"])
        self.overlap = np.array(df["Overlap"])
        self.transform = transform

    def __getitem__(self, idx):
        reference_timestamp = self.reference_timestamps[idx]
        reference_kf = KeyFrame(directory=self.root_dir, scan_time=reference_timestamp)
        reference_kf.load_pointcloud()
        reference_pcd, reference_features = reference_kf.training_preprocess()

        other_timestamp = self.other_timestamps[idx]
        other_kf = KeyFrame(directory=self.root_dir, scan_time=other_timestamp)
        other_kf.load_pointcloud()
        other_pcd, other_features = other_kf.training_preprocess()
        diferencia = 1 - np.array([self.overlap[idx]])
        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return reference_pcd, reference_features, other_pcd, other_features, diferencia

    def __len__(self):
        return len(self.overlap)
        # return 10000

class GroundTruthDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.groundtruth_path
        # Prepare data
        euroc_read = EurocReader(directory=self.root_dir)
        self.scan_times, self.pos, _, _ = euroc_read.prepare_experimental_data(
            deltaxy=EXP_PARAMETERS.exp_deltaxy,
            deltath=EXP_PARAMETERS.exp_deltath,
            nmax_scans=EXP_PARAMETERS.exp_long,
            gps_mode='utm')

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        pcd, features = kf.training_preprocess()

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return pcd, features, position

    def __len__(self):
        return len(self.scan_times)


class ValidationDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.validation_path
        # Prepare data
        euroc_read = EurocReader(directory=self.root_dir)
        self.scan_times, self.pos = euroc_read.prepare_gps_data(deltaxy=5,
            gps_mode='utm')

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
        position = np.reshape(position, (1, 3))
        kf = KeyFrame(directory=self.root_dir, scan_time=timestamp)
        kf.load_pointcloud()
        pcd, features = kf.training_preprocess()

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
    def __init__(self, in_feat, out_feat, D):
        super(VGG16_3DNetwork, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
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
            ME.MinkowskiLinear(512, out_feat))

    def forward(self, x):
        x = x.sparse()
        embedding = self.net(x).F
        if TRAINING_PARAMETERS.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings
        return embedding

    # def forward(self, input1, input2):
    #     output1 = self.forward_once(input1)
    #     output2 = self.forward_once(input2)
    #     return output1, output2


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

def get_latent_vectors(dataloader, model, main_device, secondary_device):
    torch.cuda.set_device(main_device)
    model.to(main_device)
    model.eval()

    all_descriptors = []
    all_poses = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            pcd, feat, poses = data
            poses = poses.to(secondary_device)

            input = ME.TensorField(
                features=feat.to(dtype=torch.float32),
                coordinates=pcd.to(dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=main_device,
            )
            batched_descriptor = model(input)
            batched_descriptor = batched_descriptor.to(secondary_device)
            if i == 0:
                all_descriptors = batched_descriptor
                all_poses = poses
                continue

            all_descriptors = torch.cat((all_descriptors, batched_descriptor), dim=0)
            all_poses = torch.vstack((all_poses, poses))

    return all_descriptors, all_poses

def compute_validation(model, validation_dataloader, groundtruth_dataloader):
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")

    querys_descriptors, querys_poses = get_latent_vectors(dataloader=validation_dataloader, model=model,
                                                                  main_device=device1, secondary_device=device3)
    map_descriptors, map_poses = get_latent_vectors(dataloader=groundtruth_dataloader, model=model,
                                                                  main_device=device2, secondary_device=device3)

    k = 0
    errors = []
    for query_descriptor in querys_descriptors:
        descriptor_space_distances = F.pairwise_distance(query_descriptor, map_descriptors, keepdim=True)
        predicted_pose = map_poses[torch.argmin(descriptor_space_distances)]
        real_pose = querys_poses[k]
        pose_error =  F.pairwise_distance(predicted_pose, real_pose, keepdim=True)
        errors.append(pose_error.detach().cpu().numpy())
        k += 1
    errors = np.array(errors)
    print(errors)
    return np.mean(errors), np.median(errors)

def visualize_trajectories():
    # Prepare data
    euroc_read_validation = EurocReader(directory=TRAINING_PARAMETERS.validation_path)
    scan_times_validation, pos_validation = euroc_read_validation.prepare_gps_data(deltaxy=5,
        gps_mode='lat_long')

    euroc_read_groundtrutn = EurocReader(directory=TRAINING_PARAMETERS.groundtruth_path)
    scan_times_groundtruth, pos_groundtruth, _, _ = euroc_read_groundtrutn.prepare_experimental_data(
        deltaxy=EXP_PARAMETERS.exp_deltaxy,
        deltath=EXP_PARAMETERS.exp_deltath,
        nmax_scans=EXP_PARAMETERS.exp_long,
        gps_mode='lat_long')

    gmap = CustomGoogleMapPlotter(EXP_PARAMETERS.origin_lat, EXP_PARAMETERS.origin_lon, zoom=20,
                                  map_type='satellite')

    gmap.pos_scatter(pos_validation[:, 0], pos_validation[:, 1], color='orange')
    gmap.pos_scatter(pos_groundtruth[:, 0], pos_groundtruth[:, 1], color='blue')
    gmap.draw(TRAINING_PARAMETERS.validation_path + '/map.html')


if __name__ == '__main__':
    visualize_trajectories()
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device0)
    # device1 = torch.device("cuda:1")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
    # print(model)
    # load data
    train_dataset = TrainingDataset()
    groudtruth_dataset = GroundTruthDataset()
    validation_dataset = ValidationDataset()
    # validation_example = ValidationExample()
    # ref_dataset = ReferenceDataset()
    # other_dataset = OtherDataset()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # ref_dataloader = DataLoader(ref_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)
    # other_dataloader = DataLoader(other_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_PARAMETERS.training_batch_size, shuffle=True,
                                  collate_fn=training_collation_fn)
    groundtruth_dataloader = DataLoader(groudtruth_dataset, batch_size=TRAINING_PARAMETERS.groundtruth_batch_size, shuffle=False,
                                  collate_fn=ground_collation_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=TRAINING_PARAMETERS.validation_batch_size, shuffle=False,
                                  collate_fn=ground_collation_fn)

    # initialize model and optimizer
    net = VGG16_3DNetwork(
        3,  # in channels
        TRAINING_PARAMETERS.output_size,  # out channels
        D=3).to(device0) # Space dimension
    # net = MinkowskiFCNN(
    #     3,  # in nchannel
    #     TRAINING_PARAMETERS.output_size,  # out_nchannel
    #     D=3).to(device0) # Space dimension
    # net = MinkowskiPointNet(
    #     3,  # in channels
    #     TRAINING_PARAMETERS.output_size,  # out channels
    #     dimension=3).to(device0)  # Space dimension
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = nn.DataParallel(net)

    # net.to(device)



    # net = STR2NETWORK['MinkowskiFCNN'](
    #     in_channel=3, out_channel=40, embedding_channel=1024
    # ).to(device)
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")


    criterion = ContrastiveLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train model
    counter = []
    loss_history = []
    iteration_number = 0
    net_name = '3DVGG16'

    for epoch in range(TRAINING_PARAMETERS.number_of_epochs):
        i = 0
        # for ref_data, other_data in zip(ref_dataloader, other_dataloader):
        for training_data in train_dataloader:
            optimizer.zero_grad()
            # Get new data
            ref_pcd, ref_feat, other_pcd, other_feat, label = training_data
            # ref_pcd, ref_feat, label = ref_data
            # other_pcd, other_feat = other_data
            # ref_input = ME.SparseTensor(features=ref_feat.to(dtype=torch.float32), coordinates=ref_pcd.to(dtype=torch.float32), device=device)
            # other_input = ME.SparseTensor(features=other_feat.to(dtype=torch.float32), coordinates=other_pcd.to(dtype=torch.float32), device=device)

            ref_input = ME.TensorField(
                features=ref_feat.to(dtype=torch.float32),
                coordinates=ref_pcd.to(dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device0,
            )

            other_input = ME.TensorField(
                features=other_feat.to(dtype=torch.float32),
                coordinates=other_pcd.to(dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device0,
            )

            label = label.to(device0)

            # Forward
            ref_desc = net(ref_input)
            other_desc = net(other_input)
            # loss = criterion(ref_desc.F, other_desc.F, label) #For sparse tensor
            loss = criterion(ref_desc, other_desc, label)  # For tensor field
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                # torch.save(net.state_dict(), 'red_prueba')
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                print("Epoch number {}\n Iteration number {}\n Current loss {}".format(epoch, iteration_number, loss.item()))
                mean_error, median_error = compute_validation(net, validation_dataloader, groundtruth_dataloader)
                print('Validation results \n Mean pose error: {} meters, Median error: {} meters \n'.format(mean_error, median_error))
                net.to(device0)
                torch.cuda.set_device(device0)
                # save trained model
                torch.save(net.state_dict(), net_name)
                net.train(mode=True)
            i += 1

    # save trained model
    # torch.save(net, net_name)


