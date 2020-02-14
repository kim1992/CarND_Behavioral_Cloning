import csv
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Conv2D, Cropping2D
from keras import backend as K
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

# Load data from driving_log.csv
def read_lines():
    samples = []
    with open('./data/driving_log.csv') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i > 0:
                samples.append(line)
    return samples

# Create the training model, based on NVIDA' model
def modeling():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # crop the top 70 rows and bottom 25 rows
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    # model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()
    return model

# Train the model
def training(model, train_generator, validation_generator):

    # record the training time
    t1 = time.time()

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_samples)/BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/BATCH_SIZE,
                    epochs=EPOCH, verbose=1)
    t2 = time.time()
    print(round(t2-t1, 2), 'Seconds to train the model...')

    # plot the loss trend
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()  

if __name__ == '__main__':
    # load the training data
    samples = read_lines()
    samples = shuffle(samples)

    # split samples into training and validation
    train_samples, validation_samples = train_test_split(samples, test_size=0.1)

    # create the training and validation generator
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    # create the model
    model = modeling()

    # train the model
    training(model, train_generator, validation_generator)

    # save the trained model
    model.save('model.h5')
