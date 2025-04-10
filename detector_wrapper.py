import torch
import torch.nn as nn
from preprocessor import DataProcessor
from vfe import  DynamicPillarVFE
from head import TransFusionHead
from height_compression import PointPillarScatter
from bev_backbone import BaseBEVBackbone
from typing import List

class Multitask(nn.Module) : 
    def __init__(self, config, point_cloud_range, voxel_size, grid_size):
        super().__init__()
        self.config = config
        grid_size = grid_size
        self.preprocessor = DataProcessor(point_cloud_range)
        self.lidar_vfe = DynamicPillarVFE(config.LIDAR_VFE, 5, voxel_size, grid_size, point_cloud_range)
        channels = self.lidar_vfe.get_output_feature_dim()
        self.lidar_map_to_bev = PointPillarScatter(config.LIDAR_MAP_TO_BEV, grid_size)
        channels = self.lidar_map_to_bev.num_bev_features
        self.backbone_2d = BaseBEVBackbone(config.BACKBONE_2D, channels)
        channels = self.backbone_2d.num_bev_features
        self.dense_head = TransFusionHead(config.DENSE_HEAD, channels, config.DENSE_HEAD.NUM_CLASSES, None , grid_size, point_cloud_range, voxel_size)
        

    def post_process(self, predictions:List[torch.Tensor]):
        boxes, scores, labels = predictions

        indexes = scores.argsort(descending=True)
        
        scores = scores[indexes]
        boxes = boxes[indexes]
        labels = labels[indexes]

        mask = scores > 0.5

        scores = scores[mask]
        boxes = boxes[mask]
        labels = labels[mask]

        return [boxes, scores, labels]

    def load_pretrained_model(self, pretrained_dict): 
        new_dict = {}
        for k,v in pretrained_dict.items() : 
            if k in self.state_dict().keys():
                new_dict[k] = v
            else : 
                if k == 'vfe' : 
                    new_dict['lidar_vfe'] = v
        
        self.load_state_dict(new_dict)

    def forward(self, points):
        points = self.preprocessor(points)
        voxel_features, voxel_coords = self.lidar_vfe(points)
        spatial_features = self.lidar_map_to_bev(voxel_features, voxel_coords)
        spatial_features = self.backbone_2d(spatial_features)
        predictions = self.dense_head(spatial_features)

        final_pred = self.post_process(predictions)
        return final_pred


