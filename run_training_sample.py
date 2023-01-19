import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import TRAINING_PARAMETERS, EXP_PARAMETERS, ICP_PARAMETERS
import pandas as pd
import numpy as np
from scan_tools.keyframe import KeyFrame
import MinkowskiEngine as ME
from examples.classification_modelnet40 import *

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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
    # print(model)
    # load data
    ref_dataset = ReferenceDataset()
    other_dataset = OtherDataset()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    ref_dataloader = DataLoader(ref_dataset, batch_size=32, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)
    other_dataloader = DataLoader(other_dataset, batch_size=32, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)

    # initialize model and optimizer
    # net = VGG16_3DNetwork(
    #     3,  # in channels
    #     16,  # out channels
    #     D=3).to(device) # Space dimension
    net = MinkowskiSplatFCNN(
        3,  # in nchannel
        16,  # out_nchannel
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
    num_epochs = 10
    counter = []
    loss_history = []
    iteration_number = 0
    net_name = 'MinkowskiSplatFCNN'
    for epoch in range(num_epochs):
        i = 0
        for ref_data, other_data in zip(ref_dataloader, other_dataloader):
            optimizer.zero_grad()
            # Get new data
            ref_pcd, ref_feat, label = ref_data
            other_pcd, other_feat, = other_data
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
            ref_desc = net(ref_input)
            other_desc = net(other_input)
            # loss = criterion(ref_desc.F, other_desc.F, label) #For sparse tensor
            loss = criterion(ref_desc, other_desc, label)  # For tensor field
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
            i += 1

    # save trained model
    torch.save(net, net_name)


