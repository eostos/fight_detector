from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
from mamonfight22 import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
from skimage.transform import resize
import sys
from skimage.transform import resize
import numpy as np
#reload(sys)
#sys.setdefaultencoding("utf-8")
np.random.seed(1234)
def mamon_videoFightModel2(tf,wight='mamonbest947oscombo.h5'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30 , return_sequences= True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model
#### import cv2
model = mamon_videoFightModel2(tf)


cap = cv2.VideoCapture(0)
i = 0
frames = np.zeros((30, 160, 160, 3), dtype=np.float)
old = []
j = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("no frame ****")
        break  # Break if there are no more frames
        

    font = cv2.FONT_HERSHEY_SIMPLEX

    if i > 29:
        print("checking ")
        ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
        ysdatav2[0][:][:] = frames
        # Assuming you have a pred_fight function defined
        predaction = pred_fight(model, ysdatav2, acuracy=0.975)
        # Perform actions based on the prediction

        if predaction[0] == True:
            cv2.putText(frame,
                        'Violence Detected... Violence.. violence',
                        (50, 50),
                        font, 3,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4)
            cv2.imshow('video2', frame)
            cv2.waitKey(1)
            print('Violence detected here...')
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #vio = cv2.VideoWriter(f"./videos/output-{j}.avi", fourcc, 10.0, (fwidth, fheight))
            #for frameinss in old:
            #    vio.write(frameinss)
            #vio.release()

        i = 0
        j += 1
        frames = np.zeros((30, 160, 160, 3), dtype=np.float)
        old = []
    else:
        # Check if frame is not None before processing
        if frame is not None:
            frm = resize(frame, (160, 160, 3))
            old.append(frame)
            fshape = frame.shape
            fheight = fshape[0]
            fwidth = fshape[1]
            
            frm = np.expand_dims(frm, axis=0)
            if np.max(frm) > 1:
                frm = frm / 255.0
            frames[i][:] = frm

            i += 1

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
#cv2.destroyAllWindows()