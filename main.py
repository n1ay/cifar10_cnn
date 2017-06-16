from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.datasets import cifar10
from keras import optimizers, losses
import keras

def main():

    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('x_train shape:', x_train.shape, '\ny_train shape:', y_train.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    model = Sequential([
        Conv2D(input_shape=(32, 32, 3), filters=16, kernel_size=5, strides=(1,1), padding='valid',
               dilation_rate=(1,1), activation='relu', data_format='channels_last'),
        Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=5, strides=(2, 2), padding='valid',
               dilation_rate=(1, 1), activation='relu', data_format='channels_last'),
        MaxPooling2D(pool_size=(2,2), data_format='channels_last'),
        Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='valid',
               dilation_rate=(1, 1), activation='relu', data_format='channels_last'),
        Dropout(0.25),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=num_classes, activation='softmax')
    ])

    optimizer = optimizers.Adadelta()
    model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=[x_test, y_test])

    predictions = model.predict(x_test)
    print(predictions.shape)



if __name__ == '__main__':
    main()