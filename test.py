import tensorflow as tf
import numpy as np
import cv2
from def_lib import detect, landmarks_to_embedding, draw_prediction_on_image, get_keypoint_landmarks

model_path = './models/model_yoga.h5'
image_path = './images/tree2.jpg'
image_pred_path = './results/pose_test.png'
image_result_path = './results/result.png'

model = tf.keras.models.load_model(model_path)

image = tf.io.read_file(image_path)
# print(image.shape)
image = tf.io.decode_jpeg(image)
# print(image)
image_height, image_width, channel = image.shape

person = detect(image)

pose_landmarks = get_keypoint_landmarks(person)

# pose_landmarks
lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51)))
class_names = ['chair', 'cobra', 'dog', 'tree', 'warrior']
predict = model.predict(lm_pose)
print("This picture is:", class_names[np.argmax(predict[0])])
print("Accuracy:", np.max(predict[0], axis=0))
acc = round(np.max(predict[0], axis=0)*100, 2)
print(class_names)
print(np.argmax(predict))

font = cv2.FONT_HERSHEY_DUPLEX
org = (10, 40)
fontScale = 1
color = (0, 255, 0)
thickness = 1

image = np.array(image)
cv2.putText(image, class_names[np.argmax(predict)] + " | " + str(acc) + "%",org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

# cv2.putText(image, class_names[np.argmax(predict)], org, font,
#                     fontScale, color, thickness, cv2.LINE_AA)

image = draw_prediction_on_image(image, person, crop_region=None,
                                 close_figure=False, keep_input_size=True)

# print(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite(image_pred_path, image)

'''--------------------------------------- SHOW IMAGE -------------------------------------------'''
# Read First Image
img1 = cv2.imread(image_path)
# Read Second Image
img2 = cv2.imread(image_pred_path)
# concatenate image Horizontally
Hori = np.concatenate((img1, img2), axis=1)
cv2.imwrite(image_result_path, Hori)

# # concatenate image Vertically
# Verti = np.concatenate((img1, img2), axis=0)

cv2.imshow('CLASSIFICATION OF YOGA POSE', Hori)

cv2.waitKey(0)
cv2.destroyAllWindows()