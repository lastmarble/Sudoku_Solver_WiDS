from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from pillow_printed_digits import printed_dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
# from keras.datasets import mnist


def load_model():
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, y_train, X_test, y_test = printed_dataset()

    # reshaping image matrix
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing
    X_train /= 255
    X_test /= 255

    # one-hot encoding
    n_classes = 10
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    
    inputShape = (X_train.shape[1], X_train.shape[2], 1)
    # sequential model with 2 hidden layers
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # second set of FC => RELU layers
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    # compiling
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training
    history = model.fit(X_train, Y_train,
                        batch_size=128, epochs=10,
                        verbose=2,
                        validation_data=(X_test, Y_test))
    data_model = model
    loss_and_accuracy = data_model.evaluate(X_test, Y_test, verbose=2)

    print("Test Loss", loss_and_accuracy[0])
    print("Test Accuracy", loss_and_accuracy[1])
    return model
