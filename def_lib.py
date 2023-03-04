import csv
import numpy as np
import pandas as pd
import os
import tempfile
import tqdm
import wget

from data import BodyPart
from movenet import Movenet
import tensorflow as tf
from tensorflow import keras

import shutil
import subprocess

from matplotlib import pyplot as plt
import utils_pose as utils


'''------------------------------------------------------ MAKE FILE CSV ---------------------------------------------------------------------'''

if('movenet_thunder.tflite' not in os.listdir()):
    wget.download('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite', 'movenet_thunder.tflite')

# Load MoveNet Thunder model
movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
    #           Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(),
                                   reset_crop_region=False)

    return person


class MoveNetPreprocessor(object):
    #     this class preprocess pose samples, it predicts keypoints on the images
    #     and save those keypoints in a csv file for the later use in the classification task
    def __init__(self, images_in_folder,
                 csvs_out_path):
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._message = []
        #       Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # self._csvs_out_folder_per_class = os.path.join(train_step, 'csv_per_pose')
        # if (self._csvs_out_folder_per_class not in os.listdir()):
        #     os.makedirs(self._csvs_out_folder_per_class)

        #             get list of pose classes
        self._pose_class_names = sorted(
            [n for n in os.listdir(images_in_folder)]
        )

    def process(self, detection_threshold=0.1):
        #             Preprocess the images in the given folder
        for pose_class_name in self._pose_class_names:
            # Paths for pose class
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        pose_class_name + '.csv')

            # Detect landmarks in each image and write it to the csv files
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file,
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL)

                # Get the list of images
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder)])

                valid_image_count = 0

                # Detect pose landmarks in each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._message.append('Skipped' + image_path + ' Invalid image')
                        continue

                    # Skip images that is not RGB
                    if image.shape[2] != 3:
                        self._message.append('Skipped' + image_path + ' Image is not in RGB')
                        continue

                    person = detect(image)

                    # Save landmarks if all landmarks above than the threshold
                    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._message.append('Skipped' + image_path + '. No pose was confidentlly detected.')
                        continue

                    valid_image_count += 1

                    # Get landmarks and scale it to the same size as the input image
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                         for keypoint in person.keypoints],
                        dtype=np.float32)

                    # writing the landmark coordinates (tọa độ) to its csv files
                    coord = pose_landmarks.flatten().astype(np.str).tolist()
                    csv_out_writer.writerow([image_name] + coord)

        # Print the error message collected during preprocessing.
        print(self._message)

        # Combine all per-csv class CSVs into a sigle csv file
        all_landmarks_df = self.all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)

    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def all_landmarks_as_dataframe(self):
        # Merging all csv for each class into a single csv file
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        class_name + '.csv')
            per_class_df = pd.read_csv(csv_out_path, header=None)

            # Add the labels
            per_class_df['class_no'] = [class_index] * len(per_class_df)
            per_class_df['class_name'] = [class_name] * len(per_class_df)

            # Append the folder name to the filename first column
            per_class_df[per_class_df.columns[0]] = class_name + '/' + per_class_df[per_class_df.columns[0]]

            if total_df is None:
                # For the first class, assign(gán) its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate(nối) each class's data into the total dataframe
                total_df = pd.concat([total_df, per_class_df], axis=0)

        list_name = [[bodypart.name + '_x', bodypart.name + '_y',
                      bodypart.name + '_score'] for bodypart in BodyPart]

        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ['file_name'] + header_name
        header_map = {total_df.columns[i]: header_name[i]
                      for i in range(len(header_name))
                      }

        total_df.rename(header_map, axis=1, inplace=True)

        return total_df


'''--------------------------------------------------------------- TRAIN --------------------------------------------------------------------'''

def load_csv(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Drop the file_name columns as you don't need it during training.
    df.drop(['file_name'], axis=1, inplace=True)
    print(df)

    # Extract(Trích xuất) the list of class names
    classes = df.pop('class_name').unique()

    # Extract the labels
    y = df.pop('class_no')

    # Convert the input features and labels into the correct format for training.
    X = df.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5

    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                        BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                       BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                      [tf.size(landmarks) // (17 * 2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)
'''--------------------------------------------------- TEST -----------------------------------------------------------------------'''

def draw_prediction_on_image(image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np


def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
      [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
        for keypoint in person.keypoints],
      dtype=np.float32)
    return pose_landmarks

'''--------------------------------------------------- CONVERT JSON -----------------------------------------------------------------------'''

def convert_json():
    # Convert model h5 to model json
    list_files = subprocess.run(["tensorflowjs_converter", "--input_format=keras", "models/model_yoga.h5", "models/tfjs_model"])

    src = 'models/tfjs_model'
    path = 'static'
    dst = 'static/tfjs_model'
    # Copy
    if path in os.listdir():
        shutil.rmtree(path)
        shutil.copytree(src, dst)
        print("Completed conversion to json!")

    return
