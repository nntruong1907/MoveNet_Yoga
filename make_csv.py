import os
from def_lib import MoveNetPreprocessor

'''
# Download yoga_pose.zip and unzip to save folder yaga_cg
wget.download('http://download.tensorflow.org/data/pose_classification/yoga_poses.zip','yoga_poses.zip')
import zipfile
with zipfile.ZipFile("yoga_poses.zip","r") as zip_ref:
    zip_ref.extractall("yoga_cg")

'''

IMAGES_ROOT = "yoga_cg"


# Make data csv
images_in_train_folder = os.path.join(IMAGES_ROOT, 'train')
csvs_out_train_path = 'data/train_data.csv'

preprocessor_train = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    csvs_out_path=csvs_out_train_path,
)

preprocessor_train.process()


images_in_test_folder = os.path.join(IMAGES_ROOT, 'test')
csvs_out_test_path = 'data/test_data.csv'

preprocessor_test = MoveNetPreprocessor(
    images_in_folder=images_in_test_folder,
    csvs_out_path=csvs_out_test_path,
)

preprocessor_test.process()