

!pip install tensorflow-gpu==2.0.0.alpha0

!pip install tqdm



!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O ./cats_and_dogs_filtered.zip


# Commented out IPython magic to ensure Python compatibility.
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %matplotlib inline
tf.__version__



dataset_path = "./cats_and_dogs_filtered.zip"

zip_object = zipfile.ZipFile(file=dataset_path, mode="r")

zip_object.extractall("./")

zip_object.close()


dataset_path_new = "./cats_and_dogs_filtered/"

train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation


IMG_SHAPE = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

# base_model.summary()


base_model.trainable = False


base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

global_average_layer

prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

"""### Defining the model"""

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

# model.summary()



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

"""### Creating Data Generators

Resizing images

    Big pre-trained architecture support only certain input sizes.

For example: MobileNet (architecture that we use) supports: (96, 96), (128, 128), (160, 160), (192, 192), (224, 224).
"""

data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")

valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")



model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)


valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

print("Accuracy after transfer learning: {}".format(valid_accuracy))






base_model.trainable = True

print("Number of layersin the base model: {}".format(len(base_model.layers)))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])



model.fit_generator(train_generator,  
                    epochs=5, 
                    validation_data=valid_generator)


valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

print("Validation accuracy after fine tuning: {}".format(valid_accuracy))

