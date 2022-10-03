import torch
import torch.nn as nn


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d((3 + args.c_in) * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        # (batch_size, num_points, k)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx


    def get_graph_feature(self, x, k=20, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = self.knn(x[:, :3], k=k)   # (batch_size, num_points, k)
            else:
                idx = self.knn(x[:, 6:], k=k)
        device_idx = x.get_device()
        device = torch.device(f'cuda:{device_idx}' if device_idx != -1 else 'cpu')

        idx_base = torch.arange(
            0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature      # (batch_size, 2*num_dims, num_points, k)

        
    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv3(x)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.get_graph_feature(x3, k=self.k)
        # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x = self.conv4(x)
        # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64+64+128+256, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv5(x)
        return x

if __name__ == "__main__":
    from types import SimpleNamespace

    device = torch.device('cuda:4')

    d = {
        'k': 40,
        'emb_dims': 1024,
        'dropout': 0.5,
        'c_in': 4
    }
    args = SimpleNamespace(**d)
    model = DGCNN(args).to(device).eval()
    print("Model created")
    B = 8
    N = 2048
    C = 4
    x = torch.empty((B, 3 + C, N), device=device).normal_(mean=0, std=0.1)
    print("X created")
    y = model(x)
    print(f"Result: {y.shape}")
