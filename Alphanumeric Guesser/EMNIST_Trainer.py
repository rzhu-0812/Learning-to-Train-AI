from tensorflow import keras
import scipy.io
import numpy as np
from sklearn.utils import class_weight

emnist_data = scipy.io.loadmat('emnist.mat')
dataset = emnist_data['dataset']

xTrain = dataset[0][0][0][0][0][0]
yTrain = dataset[0][0][0][0][0][1]
xTest = dataset[0][0][1][0][0][0]
yTest = dataset[0][0][1][0][0][1]

mapping = dataset[0][0][2]
num_classes = len(mapping)

xTrain = xTrain.astype(np.float32) / 255.0
xTest = xTest.astype(np.float32) / 255.0

def correct_orientation(images):
  return images.reshape(-1, 28, 28, 1)

xTrain = correct_orientation(xTrain)
xTest = correct_orientation(xTest)

yTrain_flat = yTrain.flatten()
class_weights = class_weight.compute_class_weight(
  'balanced',
  classes=np.unique(yTrain_flat),
  y=yTrain_flat
)
class_weights_dict = dict(enumerate(class_weights))

datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range=10,
  zoom_range=0.1,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  fill_mode='nearest'
)
datagen.fit(xTrain)

def buildModel():
  model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'), 
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
  
    keras.layers.Dense(num_classes, activation='softmax')
  ])

  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  return model

reduceLR = keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.5,
  patience=4,
  min_lr=1e-6,
  verbose=1
)

earlyStopping = keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=10,
  restore_best_weights=True,
  verbose=1
)

model = buildModel()
model.fit(
  datagen.flow(xTrain, yTrain, batch_size=128),
  epochs=60,
  validation_data=(xTest, yTest),
  callbacks=[reduceLR, earlyStopping],
  class_weight=class_weights_dict
)

testLoss, testAcc = model.evaluate(xTest, yTest, verbose=2)
print(f"Test Accuracy: {testAcc:.4f}%")

model.save('emnistModel.keras')