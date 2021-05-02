import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
data_dir=pathlib.Path("C:\\Users\\KIIT\\Desktop\\covid19_prediction_using_ct_scan_images")

image_count = len(list(data_dir.glob('*/*.png'))) + len(list(data_dir.glob('*/*.jpg')))
print(image_count)

covid = list(data_dir.glob('CT_COVID/*'))
noncovid = list(data_dir.glob('CT_NonCOVID/*'))

Batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,validation_split=0.2,subset="training",seed=123,image_size=(img_height, img_width),batch_size=Batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=Batch_size)

class_names = train_ds.class_names
print(class_names)

num_classes = 2

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.save('saved_model/my_model')