# Example 4.9: CNN experiments on MNIST
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def build_model(filters, kernel_size):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# Original model
model_base = build_model(32, (5, 5))
history_base = model_base.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=5, batch_size=200, verbose=0)
loss_base, acc_base = model_base.evaluate(X_test, y_test, verbose=0)

# Change kernel size from 32 to 64
model_64 = build_model(64, (5, 5))
history_64 = model_64.fit(X_train, y_train, validation_data=(X_test, y_test),
                          epochs=5, batch_size=200, verbose=0)
loss_64, acc_64 = model_64.evaluate(X_test, y_test, verbose=0)

# Change filter size from (5,5) to (3,3)
model_3x3 = build_model(32, (3, 3))
history_3x3 = model_3x3.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=5, batch_size=200, verbose=0)
loss_3x3, acc_3x3 = model_3x3.evaluate(X_test, y_test, verbose=0)

print('Base model (32 filters, 5x5): loss=%.4f accuracy=%.4f' % (loss_base, acc_base))
print('64 filters, 5x5: loss=%.4f accuracy=%.4f' % (loss_64, acc_64))
print('32 filters, 3x3: loss=%.4f accuracy=%.4f' % (loss_3x3, acc_3x3))

print('\nComparison:')
print('Increase filters to 64: accuracy difference = %.4f' % (acc_64 - acc_base))
print('Change kernel to 3x3: accuracy difference = %.4f' % (acc_3x3 - acc_base))
