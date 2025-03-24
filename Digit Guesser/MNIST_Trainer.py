from tensorflow import keras

mnist = keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0
xTrain = xTrain.reshape(-1, 28, 28, 1)
xTest = xTest.reshape(-1, 28, 28, 1)

datagen = keras.preprocessing.image.ImageDataGenerator(
  rotation_range = 15,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  zoom_range = 0.2,
  shear_range = 0.1,
  fill_mode = 'nearest'
)
datagen.fit(xTrain)

def buildModel():
  model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding='same', input_shape = (28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = 'softmax')
  ])
  
  model.compile(
      optimizer = keras.optimizers.Adam(learning_rate = 0.001),
      loss = 'sparse_categorical_crossentropy',
      metrics = ['accuracy']
  )
  
  return model

reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

model = buildModel()
model.fit(
  datagen.flow(xTrain, yTrain, batch_size = 64),
  epochs = 50,
  validation_data = (xTest, yTest),
  callbacks = [reduceLR, earlyStopping]
)

testLoss, testAcc = model.evaluate(xTest, yTest, verbose=2)
print(f"Test accuracy: {testAcc:.4f}")

model.save('mnistModel.keras')