#!/usr/bin/env python
"""
    File Name   :   MinkowskiEngine-multigpu_ddp
    date        :   16/12/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import os
import argparse
import numpy as np
from time import time
from urllib.request import urlretrieve
import open3d as o3d
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.multiprocessing as mp
import torch.distributed as dist
from run_multigpu_training import *

import MinkowskiEngine as ME
from scripts.examples.minkunet import MinkUNet34C
from scripts.examples.classification_modelnet40 import *


if not os.path.isfile("weights.pth"):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_ngpu", type=int, default=2)

cache = {}
min_time = np.inf


def load_file(file_name, voxel_size):
    if file_name not in cache:
        pcd = o3d.io.read_point_cloud(file_name)
        cache[file_name] = pcd

    pcd = cache[file_name]
    quantized_coords, feats = ME.utils.sparse_quantize(
        np.array(pcd.points, dtype=np.float32),
        np.array(pcd.colors, dtype=np.float32),
        quantization_size=voxel_size,
    )
    random_labels = torch.zeros(len(feats))

    return quantized_coords, feats, random_labels

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
                out_channels=256,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(256), #ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels=512,
            #     out_channels=512,
            #     kernel_size=3,
            #     stride=1,
            #     dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels=512,
            #     out_channels=512,
            #     kernel_size=3,
            #     stride=1,
            #     dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            # ME.MinkowskiConvolution(
            #     in_channels=512,
            #     out_channels=512,
            #     kernel_size=3,
            #     stride=1,
            #     dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels=512,
            #     out_channels=512,
            #     kernel_size=3,
            #     stride=1,
            #     dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels=512,
            #     out_channels=512,
            #     kernel_size=3,
            #     stride=1,
            #     dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU(),
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            ME.MinkowskiGlobalPooling(),
            ME.MinkowskiLinear(256, out_feat))

    def forward(self, x):
        return self.net(x)

def main():
    # loss and network
    config = parser.parse_args()
    num_devices = torch.cuda.device_count()
    num_devices = min(config.max_ngpu, num_devices)
    print(
        "Testing ",
        num_devices,
        " GPUs. Total batch size: ",
        num_devices * config.batch_size,
    )

    config.world_size = num_devices
    mp.spawn(main_worker, nprocs=num_devices, args=(num_devices, config))


def main_worker(gpu, ngpus_per_node, args):
    global min_time
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.rank = 0 * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        world_size=args.world_size,
        rank=args.rank,
    )
    # create model
    model = MinkUNet34C(3, 20, D=3)
    # model = VGG16_3DNetwork(
    #     3,  # in channels
    #     16,  # out channels
    #     D=3) # Space dimension
    # model = MinkowskiFCNN(
    #     3,  # in nchannel
    #     20,  # out_nchannel
    #     D=3)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = ContrastiveLoss().cuda(args.gpu)
    # Synchronized batch norm
    net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    # net = model
    optimizer = SGD(net.parameters(), lr=1e-1)
    print(model)
    # print(model.parameters())
    # print(net)

    train_dataset = TrainingDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_PARAMETERS.batch_size, shuffle=True,
                                  collate_fn=training_collation_fn)
    torch.autograd.set_detect_anomaly(True)
    # for iteration in range(10):
    for iteration, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()

        # Get new data
        # inputs, labels = [], []
        batch = [load_file(args.file_name, 0.05) for _ in range(args.batch_size)]
        coordinates_, featrues_, random_labels = list(zip(*batch))
        # coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)
        # inputs = ME.SparseTensor(features, coordinates, device=args.gpu)
        labels = torch.cat(random_labels).long().to(args.gpu)
        ref_pcd, ref_feat, other_pcd, other_feat, label = data
        # ref_input = ME.SparseTensor(ref_feat, ref_pcd, device=args.gpu)
        # other_input = ME.SparseTensor(other_feat, other_pcd, device=args.gpu)
        ref_input = ME.TensorField(
            features=ref_feat.to(dtype=torch.float32),
            coordinates=ref_pcd.to(dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=args.gpu,
        )
        other_input = ME.TensorField(
            features=other_feat.to(dtype=torch.float32),
            coordinates=other_pcd.to(dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=args.gpu,
        )
        # label = label.long().to(args.gpu)
        # Forward
        # ref_desc = net(ref_input)
        # other_desc = net(other_input)
        # loss = criterion(ref_desc.F, other_desc.F, label)
        # loss = criterion(ref_desc, other_desc, label)  # For tensor field
        # The raw version of the parallel_apply
        st = time()
        outputs = net(ref_input.sparse())
        # Extract features from the sparse tensors to use a pytorch criterion
        out_features = outputs.F
        loss = criterion(out_features, labels)
        # torch.save(loss, 'loss_prueba')
        # loss = torch.load('loss_prueba')
        # loss = criterion(out_features, out_features)
        # Gradient
        loss.backward()
        optimizer.step()

        t = torch.tensor(time() - st, dtype=torch.float).cuda(args.gpu)
        dist.all_reduce(t)
        min_time = min(t.detach().cpu().numpy() / ngpus_per_node, min_time)
        print(
            f"Iteration: {iteration}, Loss: {loss.item()}, Time: {t.detach().item()}, Min time: {min_time}"
        )

        # Must clear cache at regular interval
        if iteration % 10 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
