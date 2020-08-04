import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import matplotlib.pyplot as plt
import glob, pylab, pandas as pd
import numpy as np
import gc

from tensorflow import keras
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Conv3D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

df_detailed = pd.read_csv('./stage_2_detailed_class_info.csv')
df_detailed['patientId'] =   df_detailed['patientId'].astype(str) + '.png'

image_gen = ImageDataGenerator(rotation_range = 30,
                              width_shift_range= 0.1,
                              height_shift_range= 0.1,
                              rescale= 1/255,
                              shear_range=0.2,
                              zoom_range= 0.2,
                              horizontal_flip= True,
                              fill_mode='nearest',
                               validation_split =0.2
                              )

train_image_gen = image_gen.flow_from_dataframe(dataframe = df_detailed,
                             directory= './train',
                             x_col = 'patientId',
                             y_col = 'class',
                              target_size=(224,224),
                              color_mode='rgb',
                              classes= None,
                              class_mode='categorical',
                              batch_size=16,
                              shuffle=True,
                              subset = 'training'
                             )

test_image_gen = image_gen.flow_from_dataframe(dataframe = df_detailed,
                             directory= './train',
                             x_col = 'patientId',
                             y_col = 'class',
                              target_size=(224,224),
                              color_mode='rgb',
                              classes= None,
                              class_mode='categorical',
                              batch_size=16,
                              shuffle=True,
                              subset = 'validation'
                             )


orig_net = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))

filters = GlobalAveragePooling2D()(orig_net.output)

classifiers = Dense(3, activation='softmax', bias_initializer='ones')(filters)

model = Model(inputs=orig_net.inputs, outputs=classifiers)

model.compile(loss='categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])



from keras.callbacks import ModelCheckpoint, EarlyStopping


checkpoint = ModelCheckpoint("dense121.h5", monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=20,
                      verbose=1,
                      mode='auto')

gc.collect()

results = model.fit_generator(train_image_gen, epochs =150,
                              steps_per_epoch=400,
                             validation_data=test_image_gen,
                              callbacks=[checkpoint,early],
                              validation_steps = 16)
