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
from examples.classification_modelnet40 import *
from eurocreader.eurocreader_outdoors import EurocReader
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

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return reference_pcd, reference_features, other_pcd, other_features, np.array([self.overlap[idx]])

    def __len__(self):
        return len(self.overlap)

class GroundTruthDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.ground_truth_path
        # Prepare data
        euroc_read = EurocReader(directory=self.root_dir)
        self.scan_times, self.pos, _, _ = euroc_read.prepare_ekf_data(
            deltaxy=EXP_PARAMETERS.exp_deltaxy,
            deltath=EXP_PARAMETERS.exp_deltath,
            nmax_scans=EXP_PARAMETERS.exp_long)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
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
        self.scan_times, self.pos, _, _ = euroc_read.prepare_ekf_data(
            deltaxy=5,
            deltath=EXP_PARAMETERS.exp_deltath,
            nmax_scans=EXP_PARAMETERS.exp_long)

    def __getitem__(self, idx):
        timestamp = self.scan_times[idx]
        position = self.pos[idx]
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
        batched_pcd = torch.zeros((N, 4), dtype=torch.int32, device=device)
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
            ME.MinkowskiLinear(512, out_feat))

    def forward_once(self, x):
        return self.net(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


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

def compute_validation(validation_dataloader, groundtruth_dataloader, net):
    all_querys_descriptors = []
    map_descriptors = []
    # for val_data in validation_dataloader:
    for i, val_data in enumerate(validation_dataloader, 0):
        querys_pcd, querys_feat, querys_poses = val_data

        input_querys = ME.TensorField(
            features=querys_feat.to(dtype=torch.float32),
            coordinates=querys_pcd.to(dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )
        batched_querys_descriptor = net(input_querys)
        if i == 0:
            all_querys_descriptors = batched_querys_descriptor
        else:
            all_querys_descriptors = torch.cat((all_querys_descriptors, batched_querys_descriptor), dim=0)


    for i, gd_data in enumerate(groundtruth_dataloader, 0):
        gd_pcd, gd_feat, gd_poses = gd_data
        gd_input = ME.TensorField(
            features=gd_feat.to(dtype=torch.float32),
            coordinates=gd_pcd.to(dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )
        submap_descriptors = net(gd_input)
        if i == 0:
            map_descriptors = submap_descriptors
        else:
            map_descriptors = torch.cat((map_descriptors, submap_descriptors), dim=0)

    for query_descriptor in all_querys_descriptors:
        query_map_distance = F.pairwise_distance(query_descriptor, map_descriptors, keepdim=True)
        print(query_map_distance)







if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
    # print(model)
    # load data
    train_dataset = TrainingDataset()
    groudtruth_dataset = GroundTruthDataset()
    validation_dataset = ValidationDataset()
    validation_example = ValidationExample()
    # ref_dataset = ReferenceDataset()
    # other_dataset = OtherDataset()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # ref_dataloader = DataLoader(ref_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)
    # other_dataloader = DataLoader(other_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True,
                                  collate_fn=training_collation_fn)
    groundtruth_dataloader = DataLoader(groudtruth_dataset, batch_size=64, shuffle=True,
                                  collate_fn=ground_collation_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True,
                                  collate_fn=ground_collation_fn)

    # initialize model and optimizer
    # net = VGG16_3DNetwork(
    #     3,  # in channels
    #     16,  # out channels
    #     D=3).to(device) # Space dimension
    net = MinkowskiFCNN(
        3,  # in nchannel
        TRAINING_PARAMETERS.output_size,  # out_nchannel
        D=3).to(device) # Space dimension
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
    net_name = 'MinkowskiFCNN'
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
                device=device,
            )

            other_input = ME.TensorField(
                features=other_feat.to(dtype=torch.float32),
                coordinates=other_pcd.to(dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device,
            )

            label = label.to(device)

            # Forward
            """
            ref_desc = net(ref_input)
            other_desc = net(other_input)
            # loss = criterion(ref_desc.F, other_desc.F, label) #For sparse tensor
            loss = criterion(ref_desc, other_desc, label)  # For tensor field
            loss.backward()
            optimizer.step()
            """
            if i % 10 == 0:
                # print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                # iteration_number += 10
                # counter.append(iteration_number)
                # loss_history.append(loss.item())
                compute_validation(validation_dataloader, groundtruth_dataloader, net)


            i += 1

    # save trained model
    torch.save(net, net_name)


