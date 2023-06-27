import torch
import torch.nn as nn
from config.config import TRAINING_PARAMETERS
import MinkowskiEngine as ME

class VGG16_3DNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, D):
        super(VGG16_3DNetwork, self).__init__()
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
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
                dimension=D), ME.MinkowskiBatchNorm(512), ME.MinkowskiReLU())
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
            # ME.MinkowskiGlobalPooling())
            # ME.MinkowskiGlobalMaxPooling())
            # ME.MinkowskiLinear(512, out_channels))
        # self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        # self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_avg_pool = ME.MinkowskiGlobalPooling()
        # self.global_GeM_pool = GeM()
    def forward(self, x):
        verbose = False
        if verbose:
            print("Input: ", x.size())

        x = x.sparse()
        out = self.backbone(x)
        # embedding = self.global_avg_pool(out).F
        # x1 = self.global_max_pool(out)
        out = self.global_avg_pool(out)
        # out = ME.cat(x1, x2)
        # out = self.global_GeM_pool(out)
        if verbose:
            print("Output: ", out.size())
        out = out.F
        if TRAINING_PARAMETERS.normalize_embeddings:
            out = torch.nn.functional.normalize(out, p=2, dim=1)  # Normalize embeddings
        return out

class VGG16_3DNetwork_mod(nn.Module):
    def __init__(self, in_channels, out_channels, D):
        super(VGG16_3DNetwork_mod, self).__init__()
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
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
                out_channels=1024,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(1024), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(1024), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                dimension=D), ME.MinkowskiBatchNorm(1024), ME.MinkowskiReLU())


        # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dilation=1, dimension=D),
        # ME.MinkowskiGlobalPooling())
        # ME.MinkowskiGlobalMaxPooling())
        # ME.MinkowskiLinear(512, out_channels))
        # self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        # self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_avg_pool = ME.MinkowskiGlobalPooling()
        # self.global_GeM_pool = GeM()

    def forward(self, x):
        verbose = False
        if verbose:
            print("Input: ", x.size())

        x = x.sparse()
        out = self.backbone(x)
        # embedding = self.global_avg_pool(out).F
        # x1 = self.global_max_pool(out)
        out = self.global_avg_pool(out)
        # out = ME.cat(x1, x2)
        # out = self.global_GeM_pool(out)
        if verbose:
            print("Output: ", out.size())
        out = out.F
        if TRAINING_PARAMETERS.normalize_embeddings:
            out = torch.nn.functional.normalize(out, p=2, dim=1)  # Normalize embeddings
        return out