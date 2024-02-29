import shutil
import argparse
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import re
import time

def train_test_split_save(args):
    dataset_name = args["data_dir"].split("/")[-2]
    cloud_pfx = "cloud_"
    cloud_ext = ".bin"
    og_cloud_dir = "velodyne/"

    label_prefix = "cloud_" #"label_sweep"
    print("Confirm that this is correct: prefix for the LIDAR data is :", label_prefix)
    time.sleep(2)
    label_ext = ".label"
    og_label_dir = "labels/"

    cloud_dir = args["data_dir"]+og_cloud_dir
    label_dir = args["data_dir"]+og_label_dir
    save_dir = args["data_dir"]

    # preconverted .bin from .pcd via Tools_RosBag2KITTI/pcd2bin 
    clouds = glob.glob(cloud_dir + cloud_pfx + "*"+ cloud_ext)
    
    validation_portion = 0.1
    test_portion = 0.1

    print("percentage of data splited as test set: ", test_portion)
    train, validation = train_test_split(clouds, test_size=validation_portion, random_state=42)
    train, test = train_test_split(clouds, test_size=test_portion, random_state=69)

    output_sequence_dir = save_dir + "sequences/"
    if os.path.exists(output_sequence_dir):
        print("output_sequence_dir already exists, we are going to remove it, which are: \n" + output_sequence_dir)
        input("Press Enter to continue...")
        shutil.rmtree(output_sequence_dir)

    os.makedirs(output_sequence_dir + "00/labels/")
    os.makedirs(output_sequence_dir + "00/" + og_cloud_dir)
    os.makedirs(output_sequence_dir + "01/labels/")
    os.makedirs(output_sequence_dir + "01/" + og_cloud_dir)
    os.makedirs(output_sequence_dir + "02/" + og_cloud_dir)



    for file in train:
        print("loading lables for file: ", file)
        ts = re.findall(r'[0-9]+', file)
        cloud_id = cloud_pfx + ts[0] + "_" + ts[1]
        shutil.copy(file, output_sequence_dir + "00/velodyne/" + cloud_id + cloud_ext)
        label = label_dir + cloud_id + label_ext
        shutil.copy(label, output_sequence_dir + "00/labels/" + cloud_id + label_ext)
        print("successfully loaded lables for training file: ", file)
    for file in validation:
        print("loading lables for file: ", file)
        ts = re.findall(r'[0-9]+', file)
        cloud_id = cloud_pfx + ts[0] + "_" + ts[1]
        shutil.copy(file, output_sequence_dir + "01/velodyne/" + cloud_id + cloud_ext)
        label = label_dir + cloud_id + label_ext
        shutil.copy(label, output_sequence_dir + "01/labels/" + cloud_id + label_ext)
        print("successfully loaded lables for validation file: ", file)
    for file in test:
        print("loading lables for file: ", file)
        ts = re.findall(r'[0-9]+', file)
        cloud_id = cloud_pfx + ts[0] + "_" + ts[1]
        shutil.copy(file, output_sequence_dir + "02/velodyne/" + cloud_id + cloud_ext)
        print("successfully loaded lables for test file: ", file)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "-D", required=True, help="The main dataset directory where the labeled data is stored with the final '/' included")
    args = vars(ap.parse_args())

    train_test_split_save(args)