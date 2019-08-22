from models.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
from keras import models, layers, optimizers, backend
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
backend.set_session(
    session=tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True))))

vgg16 = VGG16(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3))
vgg16.summary()

for layer in vgg16.layers:
    layer.trainable = False

model = models.Sequential()
model.add(vgg16)

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu', name='fc1'))
model.add(layers.Dense(4096, activation='relu', name='fc2'))
model.add(layers.Lambda(lambda x: backend.l2_normalize(x, axis=1)))
model.add(layers.Dense(2, activation='softmax', name='predictions'))

model.summary()
for layer in model.layers:
    print(layer, layer.trainable)

train_dir = "../../data/data_a2/train"
validation_dir = "../../data/data_a2/validation"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True)
# fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 64
val_batchsize = 16
image_size = 224

# data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# callbacks
loss_checkpointer = ModelCheckpoint(
    filepath="../../weights/a2_best_model.hdf5",
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
tensorboard_callback = TensorBoard(
    log_dir='./graph',
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1,
    callbacks=[loss_checkpointer, early_stopping, tensorboard_callback])

# Save the model
model.save('../../weights/a2_model_15_epochs.h5')
