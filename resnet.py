import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import data as tf_data
from keras.callbacks import CSVLogger

# Load pretrained ResNet50 
# without top classifier (output),
# stored in ~/.keras/models/

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

print(base_model.summary())
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)

# Create output layer
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Allow training on
# convolutional layer (5)

for layer in base_model.layers:
    if "conv4" in layer.name:
        layer.trainable = True
    if "conv5" in layer.name:
        layer.trainable = True

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ],
)

# Create training and validation sets
train_images = "data/HARIS_classification_train/"
validation_images = "data/HARIS_classification_validation/"

image_size = (224, 224)
batch_size = 64

train_ds = keras.utils.image_dataset_from_directory(
    train_images,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    validation_images,
    image_size=image_size,
    batch_size=batch_size,
)

# Augment images
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

# Prefetching samples in GPU memory 
# helps maximize GPU utilization.

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# Log results to CSV file
csv_logger = CSVLogger("results.csv", append=True)

callbacks = [
    keras.callbacks.ModelCheckpoint("saves/save_at_{epoch}.keras"),
    csv_logger
]

epochs = 50
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
