import os

import numpy as np
import torch
import torch.nn.functional as F
from lib.pointops2.functions import pointops
from scipy.linalg import svd
from torch_geometric.nn import voxel_grid


def grid_sample(pos, batch, size, start, return_p2v=True):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None

    cluster = voxel_grid(pos, batch, size, start=start)  # [N, ]

    if return_p2v == False:
        unique, cluster = torch.unique(
            cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(
        cluster, sorted=True, return_inverse=True, return_counts=True)

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k)  # [n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1)  # [n, k]
    p2v_map[mask] = torch.argsort(cluster)

    return cluster, p2v_map, counts


def compute_hog_deprecated(x, k, use_cpu=False):
    '''
    Compute histogram of oriented gradients using cell size of 1
    so that every point gets information of its neighbors
    x (B x 3 x N)
    k (number of nbrs to consider)
    '''
    batch_size = x.size(0)
    num_pts = x.size(2)

    nn_idx = knn(x, k).view(-1)
    # B x N x k x 3
    x_nn = x.contiguous().view(batch_size * num_pts, -
                               1)[nn_idx, :].view(batch_size,
                                                  num_pts, k, 3)
    # center the pointcloud
    mean = x_nn.mean(dim=2, keepdim=True)  # B x N x 1 x 3
    centered = x_nn - mean
    # perform svd to obtain gradients & magnitudes
    # considering s as mag because |v|=1 for all points
    _, s, v = svd(centered.detach().cpu().numpy(),
                  full_matrices=False, overwrite_a=True, check_finite=False)
    # convert to tensors
    v = torch.from_numpy(v)
    s = torch.from_numpy(np.sqrt(s))
    # move to appropriate device
    if "LOCAL_RANK" in os.environ:
        v = v.cuda(int(os.environ["LOCAL_RANK"]))
        s = s.cuda(int(os.environ["LOCAL_RANK"]))
    elif not use_cpu:
        v = v.cuda()
        s = s.cuda()
    # get the first element (largest variance)
    gradients = v[:, :, 0]  # BxNx3x3 -> BxNx3
    magnitudes = s[:, :, 0].unsqueeze(-1)  # BxNx3 -> BxNx1

    # orient grads and mags into knn shape
    gradients_nn = gradients.view(
        batch_size * num_pts, -1)[nn_idx, :].view(batch_size, num_pts, k, 3)
    magnitudes_nn = magnitudes.view(
        batch_size * num_pts, -1)[nn_idx, :].view(batch_size, num_pts, k, 1)
    # compute angles
    zenith = torch.acos(gradients_nn[:, :, :, 2]).unsqueeze(-1) * 180 / np.pi
    azimuth = torch.atan(
        gradients_nn[:, :, :, 1] / gradients_nn[:, :, :, 0]).unsqueeze(-1) * 180 / np.pi
    # stack into cells (zenith, azimuth, magnitude) (b, n, k, 3)
    cells = torch.cat((zenith.int(), azimuth.int(), magnitudes_nn), dim=-1)
    # don't differentiate between signed and unsigned
    cells[cells < 0] += 180
    # init histogram
    if use_cpu:
        histogram = torch.zeros((batch_size, num_pts, 9, 2))
    else:
        if 'LOCAL_RANK' in os.environ:
            histogram = torch.zeros((batch_size, num_pts, 9, 2), device=torch.device(
                int(os.environ['LOCAL_RANK'])))
        else:
            histogram = torch.zeros(
                (batch_size, num_pts, 9, 2), device=torch.device('cuda'))
    # 20 degrees bins computed from angles
    bins = torch.floor(cells[:, :, :, 2] / 20.0 - 0.5) % 9
    # vote for bin i
    width = 20.0
    num_bins = 9
    first_centers = width * ((bins + 1) % num_bins + 0.5)
    first_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((first_centers - cells[:, :, :, :2]) % 180) / width
    # vote for next bin
    second_centers = width * (bins + 0.5)
    second_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((cells[:, :, :, :2] - second_centers) % 180) / width
    for c in range(9):
        histogram[:, :, c] += (first_votes * (bins == c)).sum(dim=2)
        histogram[:, :, (c+1) % 9] += (second_votes * (bins == c)).sum(dim=2)
    histogram = F.normalize(histogram, p=2.0, dim=2)
    histogram = histogram.view(batch_size, num_pts, -1)
    return histogram


def compute_hog(xyz, window_size, offset, batch, window_maps=None, k=40, choice="knn", radius=0.1):
    '''
    xyz: (n, 3) = coordinates of all points
    window_size: float = size of window
    offset: (batch_size) = batch offset
    batch: (n, ) = index of the batch that the ith point belongs to
    window_maps: (2, ) = v2p_map and p2v_map 
    k: int = number of nearest neighbors
    choice: str = "knn" or "ballquery"
    radius: float = radius of ball query
    
    Histogram of Oriented Gradients in Non-Overlapping Cubic Windows
    
    1. Query neighbors of all points
    2. Reduce each neighborhood by applying SVD
    3. Values obtained are s (magnitude) and v[0] (gradient)
    4. get angles from gradients & convert angles to unsigned
    5. initialize histogram for each cubic window as 2D (zenith & azimuth) array of 9 cells (20 degrees each)
    6. each window contributes its gradient to the respective cell of the histogram as per its angles
    8. (optional) normalize gradients of each window by
        (option 1) considering that window only
        (option 2) considering 3 adjacent windows
    9. flatten the histogram and return it (n, 18)
    '''
    # p2v_map: (num windows, max points per window) index of a point and the window that it has been mapped to
    # v2p_map: (n) index of window that ith point has been mapped to
    if window_maps is None:
        v2p_map, p2v_map, counts = grid_sample(xyz, batch, window_size, start=None)
    else:
        v2p_map, p2v_map = window_maps
    # v2p_map = v2p_map.cpu()
    # p2v_map = p2v_map.cpu()
    num_pts = xyz.size(0)
    device = xyz.get_device()
    # 1. Query neighbors of all points & get displacement to neighbors
    if choice == "knn":
        # neighbor_idx: (n, k) = indices of all points & their k nearest neighbors
        neighbor_idx = pointops.knnquery(k, xyz, xyz, offset, offset)[
            0].contiguous()
    elif choice == "ballquery":
        raise NotImplementedError("Ball Query not yet supported")
        # neighbor_idx = tp.ball_query(radius, k, xyz, xyz, mode="partial_dense", batch_x=batch, batch_y=batch)[0].contiguous()
    else:
        raise NotImplementedError(
            "Invalid option for choice of nearest neighbors")
    # (n, 3)
    mean_per_nbrhood = torch.index_select(
        xyz, dim=0, index=neighbor_idx.flatten()).view(-1, 40, 3).mean(dim=1)
    # center the data
    xyz = xyz - mean_per_nbrhood
    # get displacement to neighbors
    disp = pointops.subtraction(xyz, xyz, neighbor_idx.int())
    # 0 distance to itself
    if choice == "ballquery":
        disp[neighbor_idx == -1] = 0
    # 2. Reduce each neighborhood by applying SVD
    _, s, v = np.linalg.svd(disp.cpu().numpy(), full_matrices=False)
    # 3. Values obtained are s (magnitude) and v[0] (gradient)
    # get the first element (largest variance)
    magnitudes = torch.from_numpy(s[:, 0]).cuda(device)  # N x 3 -> N
    gradients = torch.from_numpy(v[:, :, 0]).cuda(device)  # N x 3 x 3 -> N x 3
    # 4. get angles from gradients & convert angles to unsigned
    zenith = torch.acos(gradients[:, 2]) * 180 / np.pi
    zenith[zenith < 0] += 180
    # add 1e-12 for safe division
    azimuth = torch.atan(
        torch.div(gradients[:, 1], gradients[:, 0] + 1e-12)) * 180 / np.pi
    azimuth[azimuth < 0] += 180
    # initialize histogram
    histogram = torch.zeros((num_pts, 18), device=f"cuda:{device}")
    # histogram = torch.zeros((num_pts, 18))
    # fill histogram with gradients based on angle values
    for i, angles in enumerate([zenith, azimuth]):
        # which bin in its respective histogram
        # does each point contribute to
        bins = (torch.floor(angles / 20.0 - 0.5) % 9).int()
        # vote for first bin
        width = 20.0
        num_bins = 9
        first_centers = width * ((bins + 1) % num_bins + 0.5)
        first_votes = magnitudes * ((first_centers - angles) % 180) / width
        # vote for second bin
        second_centers = width * (bins + 0.5)
        second_votes = magnitudes * ((angles - second_centers) % 180) / width
        first_bin_sums = torch.zeros(
            (p2v_map.size(0), 9), device=f"cuda:{device}")
        second_bin_sums = torch.zeros(
            (p2v_map.size(0), 9), device=f"cuda:{device}")
        for c in range(9):
            # sum the total contribution of magnitudes to this bin
            first_bin_contributors = first_votes * (bins == c)
            second_bin_contributors = second_votes * (bins == c)
            # split into sum per window (num_windows, )
            first_bin_sum = torch.index_select(first_bin_contributors,
                                               dim=0,
                                               index=p2v_map.flatten()).reshape(p2v_map.shape).sum(dim=1)
            second_bin_sum = torch.index_select(second_bin_contributors,
                                                dim=0,
                                                index=p2v_map.flatten()).reshape(p2v_map.shape).sum(dim=1)
            # remove redundant contribution of index 0
            num_zeros_per_window = (p2v_map == 0).sum(dim=1)
            if first_bin_contributors[0] > 0:
                first_bin_sum = first_bin_sum - \
                    first_bin_contributors[0] * num_zeros_per_window
                # add back index 0's actual contribution
                first_bin_sum[v2p_map[0]] += first_bin_contributors[0]
            if second_bin_contributors[0] > 0:
                second_bin_sum = second_bin_sum - \
                    second_bin_contributors[0] * num_zeros_per_window
                second_bin_sum[v2p_map[0]] += second_bin_contributors[0]
            first_bin_sums[:, c] = first_bin_sum
            second_bin_sums[:, c] = second_bin_sum
        # normalize
        first_bin_sums = F.normalize(first_bin_sums, p=2.0, dim=1)
        second_bin_sums = F.normalize(second_bin_sums, p=2.0, dim=1)
        # add to histogram
        for c in range(9):
            if i == 0:
                histogram[:, c] += torch.gather(first_bin_sums[:, c],
                                                dim=0, index=v2p_map)
                histogram[:, (c + 1 % 9)] += torch.gather(second_bin_sums[:,
                                                                          c], dim=0, index=v2p_map)
            else:
                histogram[:, c +
                          9] += torch.gather(first_bin_sums[:, c], dim=0, index=v2p_map)
                histogram[:, (c + 1) % 9 + 9] += torch.gather(
                    second_bin_sums[:, c], dim=0, index=v2p_map)
    return histogram
