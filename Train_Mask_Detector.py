# importing Modules
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import os

# Initialising the parameters
INIT_LR = 1e-4
EPOCHS = 15
BATCH_SIZE = 32

# Set the working path
DIRECTORY = r'D:\Coursera\MLProject\Face Mask Detector\Datasets'
CATEGORIES = ['with_mask', 'without_mask']

print('Loading Images....')

data = []
labels =[]

# Loading the images from the pathway
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype='float32')
labels = np.array(labels)

# Split the data in train and test sets
(train_X, test_X, train_Y, test_Y)= train_test_split(data, labels, test_size= 0.2, stratify= labels, random_state= 42)

# Creating more data from the initial data
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# initialising Transfer learning
basemodel= MobileNetV2(weights='imagenet', include_top= False,
                       input_tensor= Input(shape=(224,224,3)))
headmodel= basemodel.output                                         # output shape is 7*7*1280
headmodel= AveragePooling2D(pool_size=(7,7))(headmodel)             # shape is 1*1*1280
headmodel= Flatten(name='flatten')(headmodel)
headmodel= Dense(128,activation='relu')(headmodel)
headmodel= Dropout(0.5)(headmodel)
headmodel= Dense(2,activation='softmax')(headmodel)

model= Model(inputs=basemodel.input, outputs=headmodel)

# Freezing the weights of the basemodel
for layer in basemodel.layers:
    layer.trainable= False

print('Compiling model....')
opt= Adam(lr=INIT_LR, decay= INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy',optimizer= opt, metrics=['accuracy'])

print('Training head....')
history= model.fit(
        aug.flow(train_X,train_Y,batch_size= BATCH_SIZE),
        steps_per_epoch=len(train_X)//BATCH_SIZE,
        validation_data=(test_X, test_Y),
        validation_steps=len(test_X)//BATCH_SIZE,
        epochs=EPOCHS)

print('Evaluating network....')
predidxs= model.predict(test_X, batch_size=BATCH_SIZE)
predidxs= np.argmax(predidxs,axis=1)

print(classification_report(test_Y.argmax(axis=1), predidxs,target_names= lb.classes_))

print('Saving the model....')
model.save('Mask_Detector.model', save_format='h5')

# Plotting the Graph
N= np.arange(0,EPOCHS)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, history.history['loss'], label='train_loss')
plt.plot(N, history.history['val_loss'], label='val_loss')
plt.plot(N, history.history['accuracy'], label='train_acc')
plt.plot(N, history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('# Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('plot.png')
