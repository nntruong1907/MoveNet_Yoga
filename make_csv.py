# python make_csv.py --source yoga_cg --data data

import argparse
import os
from def_lib import MoveNetPreprocessor

'''
# Download yoga_pose.zip and unzip to save folder yaga_cg
wget.download('http://download.tensorflow.org/data/pose_classification/yoga_poses.zip','yoga_poses.zip')
import zipfile
with zipfile.ZipFile("yoga_poses.zip","r") as zip_ref:
    zip_ref.extractall("yoga_cg")

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('', type=str, default='yolo7.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--source','-s', type=str, default='yoga_cg', help='folder containing data for some yoga poses')
    parser.add_argument('--data','-d', type=str, default='data', help='folder save file train and test')
    opt = parser.parse_args()
    
    folder_csv_path = opt.data
    IMAGES_ROOT = opt.source

    # Make data csv
    images_in_train_folder = os.path.join(IMAGES_ROOT, 'train')
    csvs_out_train_path = folder_csv_path + '/' + 'train_data.csv'

    preprocessor_train = MoveNetPreprocessor(
        images_in_folder=images_in_train_folder,
        csvs_out_path=csvs_out_train_path,
    )

    preprocessor_train.process()


    images_in_test_folder = os.path.join(IMAGES_ROOT, 'test')
    csvs_out_test_path = folder_csv_path + '/' + 'test_data.csv'

    preprocessor_test = MoveNetPreprocessor(
        images_in_folder=images_in_test_folder,
        csvs_out_path=csvs_out_test_path,
    )

    preprocessor_test.process()