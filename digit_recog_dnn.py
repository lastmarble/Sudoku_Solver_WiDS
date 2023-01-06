import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


def predict(test_data_x, samples):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print("X_train shape", X_train.shape)
    # print("y_train shape", y_train.shape)
    # print("X_test shape", X_test.shape)
    # print("y_test shape", y_test.shape)

    # reshaping image matrix
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    test_data_x = test_data_x.reshape(samples, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    test_data_x = test_data_x.astype('float32')

    # normalizing
    X_train /= 255
    X_test /= 255
    test_data_x /= 255

    # print("Train matrix shape", X_train.shape)
    # print("Test matrix shape", X_test.shape)

    # one-hot encoding
    n_classes = 10
    # print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    # print("Shape after one-hot encoding: ", Y_train.shape)

    # sequential model with 2 hidden layers
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compiling
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training
    history = model.fit(X_train, Y_train,
                        batch_size=128, epochs=10,
                        verbose=2,
                        validation_data=(X_test, Y_test))
    mnist_model = model
    loss_and_accuracy = mnist_model.evaluate(X_test, Y_test, verbose=2)

    print("Test Loss", loss_and_accuracy[0])
    print("Test Accuracy", loss_and_accuracy[1])

    # predicting
    predictions = np.argmax(model.predict(test_data_x), axis=-1)
    print("Prediction: ", predictions, "\nTest: ", y_test)


def predict_image(char):
    import cv2
    import imutils
    image = cv2.imread(char)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # BGR -> RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(img, width=28)
    img = img.reshape(1, 28, 28, 1)
    predict(img, 1)
