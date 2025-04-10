import torch.nn as nn 
import torch
from typing import Optional, Tuple

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,) -> torch.Tensor:
    index = broadcast(index, src, dim)
    num_pts = index.max() + 1
    dimension = src.size(1)
    out = torch.zeros(num_pts, dimension, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def scatter_sum_index(src: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int = -1,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out)
    dim_size, _ = out.shape
    index_dim = dim
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum_index(ones, index, dim_size, index_dim, None)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    out.true_divide_(count)
    return out


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = scatter_mean(x, unq_inv, dim=0)

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__()

        self.use_norm = model_cfg.USE_NORM
        self.with_distance = model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = model_cfg.USE_ABSLOTE_XYZ
        self.double_flip = model_cfg.get('DOUBLE_FLIP', False)
        self.labels = model_cfg.get('SCATTER_LABELS', False)
        self.is_radar = model_cfg.get('IS_RADAR', False)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = int(grid_size[0] * grid_size[1])
        self.scale_y = int(grid_size[1])
        
        self.grid_size = grid_size.clone().detach().cuda()
        self.voxel_size = voxel_size.cuda()
        self.point_cloud_range = point_cloud_range.cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, points):
        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).long()
        points_xyz = points[:, [1, 2,3]].contiguous()

        merge_coords = points[:, 0].long() * self.scale_xy + \
                    points_coords[:, 0] * self.scale_y + \
                    points_coords[:, 1]

        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)
        
        points_mean = scatter_mean(points_xyz, unq_inv, dim=0)

        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [points[:, 1:], f_cluster, f_center]
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.long()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                (unq_coords % self.scale_xy) // self.scale_y,
                                unq_coords % self.scale_y,
                                torch.zeros(unq_coords.shape[0]).to(unq_coords.device).long()
                                ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        ## append 
        # voxels.append(features)
        # coors.append(voxel_coords)
        # pc_voxel_ids.append(unq_inv)
        
        # voxels = torch.cat(voxels, dim=0)
        # coors = torch.cat(coors, dim=0)
        # pc_voxel_ids = torch.cat(pc_voxel_ids, dim=0)

        return features, voxel_coords

