import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import random_split
from config import TRAINING_PARAMETERS, EXP_PARAMETERS, ICP_PARAMETERS
import pandas as pd
import numpy as np
import open3d as o3d
from scan_tools.keyframe import KeyFrame

class PCDDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = TRAINING_PARAMETERS.directory
        labels_dir = TRAINING_PARAMETERS.directory + '/labelling.csv'
        self.scans_dir = TRAINING_PARAMETERS.directory + '/robot0/lidar/data/'

        df = pd.read_csv(labels_dir)
        self.reference_timestamps = np.array(df["Reference timestamp"])
        self.other_timestamps = np.array(df["Other timestamp"])
        self.overlap = np.array(df["Overlap"])
        self.transform = transform

    def __getitem__(self, idx):

        reference_timestamp = self.reference_timestamps[idx]
        reference_kf = KeyFrame(directory=self.root_dir, scan_time=reference_timestamp)
        reference_kf.load_pointcloud()
        reference_pcd = reference_kf.training_preprocess()
        # reference_pcd = o3d.io.read_point_cloud(self.scans_dir + str(reference_timestamp) + '.pcd')
        # reference_pcd = torch.from_numpy(np.asarray(reference_pcd.points))

        other_timestamp = self.other_timestamps[idx]
        other_kf = KeyFrame(directory=self.root_dir, scan_time=other_timestamp)
        other_kf.load_pointcloud()
        other_pcd = other_kf.training_preprocess()

        # other_timestamp = self.other_timestamps[idx]
        # other_pcd = o3d.io.read_point_cloud(self.scans_dir + str(other_timestamp) + '.pcd')
        # other_pcd = torch.from_numpy(np.asarray(other_pcd.points))

        # if self.transform:
        #     pointcloud = self.transform(pointcloud)
        return reference_pcd, other_pcd, torch.from_numpy(np.array([self.overlap[idx]], dtype=np.float32))

    def __len__(self):
        return len(self.overlap)

class PointCloudCNN(nn.Module):
    def __init__(self):
        super(PointCloudCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.fc = nn.Linear(256, 10)

    def forward_once(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 256)
        x = self.fc(x)

    def forward(self, x):



        return x

# load data
dataset = PCDDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# initialize model and optimizer
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        reference_pcd, other_pcd, overlap = data
#         optimizer.zero_grad()
#         outputs = model(pointclouds)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # evaluate model on test set
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_dataloader:
#             pointclouds, labels = data
#             outputs = model(pointclouds)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Epoch: {} Test Accuracy: {}%'.format(epoch + 1, 100 * correct / total))
#
# # save trained model
# torch.save(model.state_dict(), 'pointcloud_cnn.pth')

