from detector_wrapper import Multitask
import torch
import numpy as np
import os 
from pypcd4 import PointCloud
import time

def load_samples() :
    foler_name = "pointcloud"
    out = []
    for elem in os.listdir(foler_name) : 
        out.append(os.path.join(foler_name, elem))
    
    return out 

def add_fifth_dim(pts):
    ## ugly, just because we had timestamp data on nuscenes.
    ## we will retrain on roadview data anyway
    pts = torch.cat([pts, torch.zeros((pts.shape[0],1), device=pts.device)], dim=1)

    return pts 

def naive_benchmark(model, sample_list):
    load_time = 0
    forward_time = 0 
    total_time = 0

    for sample in sample_list : 
        t0 = time.time()
        points = PointCloud.from_path(sample).numpy().astype(np.float16)
        points = torch.from_numpy(points).cuda()
        points = add_fifth_dim(points)
        points = points.half()
        t1 = time.time()

        out = model(points)
        t2 = time.time()

        load_time += t1-t0
        forward_time += t2 - t1 
        total_time += t2-t0
    
    mean_load_time = load_time / len(sample_list)
    mean_forward_time = forward_time / len(sample_list) 
    mean_total_time = total_time / len(sample_list) 

    out_str = "----- Benchmark on local ORIN device ----- \n"
    out_str += f"Load time : {mean_load_time} \nForward time : {mean_forward_time} \nTotal time : {mean_total_time} \nFPS (with loading) : {np.round(1/mean_total_time)} \nFPS (inference) {np.round(1/mean_forward_time)}"
    out_str += "\n------------------------------------------"

    return out_str

if __name__ == '__main__' : 
    from load_config import load_config
    cfg = load_config('config/model_cfg.yaml')
    samples = load_samples()
    
    points = PointCloud.from_path(samples[0]).numpy()
    points = torch.from_numpy(points).cuda()
    points = add_fifth_dim(points).half()

    point_cloud_range = torch.from_numpy(np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
    voxel_size = torch.from_numpy(np.array([0.3,0.3,8]))
    grid_size = torch.from_numpy(np.array([360,360,1]))
    model = Multitask(cfg.MODEL, point_cloud_range, voxel_size, grid_size).cuda()
    model.half()
    model.eval()

    ## load model state_dict
    state_dict = torch.load('checkpoint/checkpoint.pth', weights_only=True)
    model.load_state_dict(state_dict)


    print(naive_benchmark(model, samples[:10]))
    
