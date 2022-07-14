import os.path
import os
import open3d as o3d
import pypcd.pypcd
from pypcd import *
import numpy as np
import shutil
original_path = "/home/tomvdon/ag_lab/labeling/bags/pcds/"
label_path = "/home/tomvdon/ag_lab/lidar-bonnetal/farm_dataset/labels_1/"
save_path = "/home/tomvdon/ag_lab/lidar-bonnetal/farm_dataset/point_clouds_1/"
labeled_files = [file for file in os.listdir(label_path) if file.endswith(".npy")]
for f in labeled_files:
    pointcloud = os.path.join(label_path, f)
    name = os.path.split(pointcloud)[-1][:-4]
    name = name + ".pcd"
    print(original_path + name)
    shutil.copy2(original_path + name, save_path + name)
    print("saved " + save_path + name)

