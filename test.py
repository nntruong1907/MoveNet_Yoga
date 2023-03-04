# python test.py --model models/model_yoga_LSTM.h5 --data images/tree3.jpg

import argparse
import tensorflow as tf
import numpy as np
import cv2
from def_lib import detect, landmarks_to_embedding, draw_prediction_on_image, get_keypoint_landmarks
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='models/model_yoga_LSTM.h5', help='model .h5')
    parser.add_argument('--data', '-d', type=str, default='images/tree3.jpg', help='image to be detected')
    opt = parser.parse_args()

    model_path = opt.model
    image_path = opt.data
    class_names = ['chair', 'cobra', 'dog', 'tree', 'warrior']

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
    predict = model.predict(lm_pose)

    print("Input shape:", lm_pose.shape)
    print("Output shape:", predict.shape)
    print("________________________________________________________________________")
    print("This picture is:", class_names[np.argmax(predict[0])])
    print("Accuracy:", np.max(predict[0], axis=0))
    acc = round(np.max(predict[0], axis=0)*100, 2)
    print("\n", class_names)
    print(np.argmax(predict))
    print(np.array(predict[0]))

    '''Draw on image'''
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (10, 40)
    fontScale = 1
    color = (19, 255, 30)
    thickness = 1

    image = np.array(image)
    cv2.putText(image, class_names[np.argmax(predict)] + " | " + str(acc) + "%",org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    # cv2.putText(image, class_names[np.argmax(predict)], org, font,
    #                     fontScale, color, thickness, cv2.LINE_AA)

    image = draw_prediction_on_image(image, person, crop_region=None,
                                    close_figure=False, keep_input_size=True)


    curr_datetime = datetime.now().strftime('%Hh%Mm%Ss %d_%m_%Y ')
    r = str(acc) + "% " + curr_datetime
    image_pred_path = './results/draw_skeleton %s.png' % r
    image_result_path = './results/result %s.png' % r
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