import itertools
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from def_lib import load_csv, landmarks_to_embedding, preprocess_data

train_path = 'data/train_data.csv'
test_path = 'data/test_data.csv'

# Load the train data
X, y, class_names = load_csv(train_path)
# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
# Load the test data
X_test, y_test, _ = load_csv(test_path)

print( "X_train:", X_train.shape)
print( "y_train:", y_train.shape)
print( "\nX_val:", X_val.shape)
print( "y_val:", y_val.shape)
print( "\nX_test:", X_test.shape)
print( "y_test:", y_test.shape)
print( "\nclass_names:", class_names)

# Pre-process data
processed_X_train = preprocess_data(X_train)
processed_X_val = preprocess_data(X_val)
processed_X_test = preprocess_data(X_test)

# Define the model
def LSTM():
    inputs = tf.keras.Input(shape=(34, 1))
    layer = keras.layers.LSTM(64, activation=tf.nn.relu6 , return_sequences=True)(inputs)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.LSTM(128, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(32, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def CNN():
    inputs = tf.keras.Input(34)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(inputs)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = LSTM()
# model = CNN()
# Add a checkpoint callback to store the checkpoint that has the highest validation accuracy.
checkpoint_path = "models/weights.best.hdf5"
model_path = 'models/model_yoga.h5'

checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

# Start training
from datetime import datetime
print('---------------------------------- TRAINING -------------------------------------------')
start = datetime.now()
history = model.fit(processed_X_train, y_train,
                    epochs=200,
                    batch_size=64,
                    validation_data=(processed_X_val, y_val),
                    callbacks=[checkpoint, earlystopping])
# Save model
model.save(model_path)

print('--------------------------------- EVALUATION -----------------------------------------')
loss_test, accuracy_test = model.evaluate(processed_X_test, y_test)
print('LOSS TEST: ', loss_test)
print("ACCURACY TEST: ", accuracy_test)

loss_train, accuracy_train = model.evaluate(processed_X_train, y_train)
print('LOSS TRAIN: ', loss_train)
print("ACCURACY TRAIN: ", accuracy_train)

duration = datetime.now() - start
print('--------------------------------- TRAINING TIME ---------------------------------------')
print('TRAINING COMPLECTED IN TIME: ', duration)

data_eval = 'LOSS TEST: '+ str(loss_test) + ' / ACCURACY TEST: ' + str(accuracy_test) \
            + '\n' + 'LOSS TRAIN: ' + str(loss_train) + ' / ACCURACY TRAIN: ' + str(accuracy_train) \
            + '\n' + 'TRAINING COMPLETION TIME: ' + str(duration)


'''--------------------------------------- STATISTC -------------------------------------------'''

# write Evaluation into txt
Name_f = 'statistics/Evaluation2.txt'

with open(Name_f, mode='w') as f:
    f.writelines(data_eval)
f.close()

# Visualize the training history to see whether you're overfitting.
image_acc_path ='statistics/model_acc.png'
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='best')
# plt.show()
plt.savefig(image_acc_path)
plt.close()

# Visualize the training history to see whether you're overfitting.
image_loss_path ='statistics/model_loss.png'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='best')
# plt.show()
plt.savefig(image_loss_path)
plt.close()

# write Classification_report and Confusion_matrix to file
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """Plots the confusion matrix."""
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  fig = plt.gcf()
  fig.set_size_inches(18.5, 10.5)
  fig.savefig(plot_confusion_matrix_path)

classification_report_path = 'statistics/Classification_report.txt'
plot_confusion_matrix_path = 'statistics/Confusion_matrix.png'

# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(processed_X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

plot_confusion_matrix(cm,
                      class_names,
                      title ='Confusion Matrix of Pose Yoga Classification Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                          y_pred_label))

with open(classification_report_path, mode='w') as f:
    f.writelines(classification_report(y_true_label,y_pred_label))
f.close()