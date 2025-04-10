#!/usr/bin/env python
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2
import time
from detector_wrapper import Multitask
from load_config import load_config

def add_fifth_dim(pts):
    ## ugly, just because we had timestamp data on nuscenes.
    ## we will retrain on roadview data anyway
    pts = torch.cat([pts, torch.zeros((pts.shape[0],1), device=pts.device)], dim=1)

    return pts 

class Subscriber:
    def __init__(self, node_name, subscriber_name):
        self.node = node_name
        self.sub = subscriber_name
        self.device = torch.device('cuda:0')
        cfg = load_config('config/model_cfg.yaml')
        point_cloud_range = torch.from_numpy(np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
        voxel_size = torch.from_numpy(np.array([0.3,0.3,8]))
        grid_size = torch.from_numpy(np.array([360,360,1]))
        self.model = Multitask(cfg.MODEL, point_cloud_range, voxel_size, grid_size)
        self.model.half()
        self.model.eval()

        state_dict = torch.load('checkpoint/checkpoint.pth', weights_only=True)
        self.model.load_state_dict(state_dict)


    def run(self):
        rospy.init_node(self.node, anonymous=True)
        rospy.Subscriber(self.sub, PointCloud2, self.detect)
        rospy.spin()

    def detect(self, pc_data):
        start = time.time_ns()
        size = len(pc_data.data) // 16
        xyzi = np.ndarray((size, 4), np.float16, pc_data.data).copy()
        data = torch.from_numpy(xyzi).to(self.device)
        data = add_fifth_dim(data)
        out = self.model(data)
        end = time.time_ns()
        elapsed = end - start
        print(f'{elapsed / 1000000} millisecond')

if __name__ == "__main__":
    sub = Subscriber('sub', 'Detector')
    sub.run()