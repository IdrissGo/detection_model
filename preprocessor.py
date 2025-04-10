import torch.nn as nn 
import torch
from typing import List

class DataProcessor(nn.Module):

    def __init__(self, point_cloud_range):
        super().__init__()
        self.point_cloud_range = point_cloud_range.tolist()

    
    def mask_points_outside_range(self, points:torch.Tensor, limit_range:List[float]) : 
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])

        return mask

    def pad_batch(self, points:torch.Tensor):
        #Adds batch value to points for spconv 
        num_points = points.size(0)
        zeros = torch.zeros((num_points, 1), dtype=points.dtype, device=points.device)
        points = torch.cat([zeros, points], dim=1)

        return points

    
    def forward(self, points) : 
        mask = self.mask_points_outside_range(points, self.point_cloud_range)
        points = points[mask]

        points = self.pad_batch(points)
        return points
