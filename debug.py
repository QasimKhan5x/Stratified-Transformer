from functools import partial

import torch
import torch.nn as nn
import torch_points_kernels as tp
from torch.utils.data import DataLoader
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv.kernels import KPConvLayer

from util.data_util import collate_fn_limit
from util.s3dis import S3DIS

dev = 'cuda:5'
cuda_idx = 5
device = torch.device(dev)
torch.cuda.set_device(device)

data_root = '/home/Pointnet_Pointnet2_pytorch/data/stanford_indoor3d'
train_transform = None
train_data = S3DIS(split='train', data_root=data_root, test_area=5, voxel_size=0.04, voxel_max=80000,
                   transform=train_transform)

batch_size = 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, 
                          drop_last=True, collate_fn=partial(collate_fn_limit, max_batch_points=200000, logger=None)
                        )

class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats


class KPConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(nn.Linear(in_channels, d_2, bias=False), FastBatchNorm1d(d_2, momentum=bn_momentum), activation)
        self.unary_2 = torch.nn.Sequential(nn.Linear(d_2, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum), activation)
        self.kpconv = KPConvLayer(d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum)
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        
        shortcut = feats
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats += shortcut
        return feats
    

emb = nn.ModuleList([
    KPConvSimpleBlock(6, 48, 0.04, sigma=1.0),
    KPConvResBlock(48, 48, 0.04, sigma=1.0)
]).cuda()
max_num_neighbors = 34

for coord, feat, target, offset in train_loader:  # (n, 3), (n, c), (n), (b)
    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = torch.cat([torch.tensor([ii]*o) for ii, o in enumerate(offset_)], 0).long()

    sigma = 1.0
    radius = 2.5 * 0.04 * sigma
    neighbor_idx = tp.ball_query(radius, max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

    coord, feat, target, offset, batch, neighbor_idx = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), \
                                        target.cuda(non_blocking=True), offset.cuda(non_blocking=True), \
                                        batch.cuda(non_blocking=True), neighbor_idx.cuda(non_blocking=True)
    assert batch.shape[0] == feat.shape[0]
    feat = torch.cat([feat, coord], 1)

    with torch.cuda.amp.autocast():
        for layer in emb:
            feat = layer(feat, coord, batch, neighbor_idx)
print(feat.shape)
