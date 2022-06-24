import datetime
import os
import tensorflow as tf
from tensorflow import keras

# Specify new dateset
new_dataset = 'datasets/dataset_20220623_121008/'

base_model = keras.models.load_model('models/ssd_mobilenet_v2_fpnlite_320x320_1.tar')
base_model.trainable = False  # To avoid messing with the preexisting model

inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)

model.save(f'trained-models/model{datetime.datetime.now().date()}')
