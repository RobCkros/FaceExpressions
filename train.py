import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

training_directory = 'data/train'
validation_directory = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=30,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # vertical_flip=True,
    # fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

#Training and Validating Data
train_generator = train_datagen.flow_from_directory(
    training_directory,
    target_size=(48, 48),
    batch_size=128,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_directory,
    target_size=(48, 48),
    batch_size=128,
    color_mode="grayscale",
    class_mode='categorical')

# Convulutional Neural Network (CNN) architecture, model layers specified
my_model = Sequential()
#Size of images specified
my_model.add(Conv2D(32, kernel_size=(3, 3),
             activation='relu', input_shape=(48, 48, 1)))
my_model.add(BatchNormalization())
my_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
#Down sampling to include less subset of data
my_model.add(MaxPooling2D(pool_size=(2, 2)))
#Used to prevent overfitting
my_model.add(Dropout(0.5))

my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.5))

my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.5))

my_model.add(Flatten())
my_model.add(Dense(512, activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Dropout(0.5))
#Hidden layer
my_model.add(Dense(7, activation='softmax'))
#Compiling the model
my_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-5), metrics=['accuracy'])

# Training the model
history = my_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)
#Saving the model
my_model.save('model_14.h5')






























