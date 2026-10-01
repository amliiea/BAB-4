#Example 4.9 Simple CNN for the MNIST Dataset
from time import perf_counter

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
for index in range(4):
	plt.subplot(141 + index)
	plt.imshow(X_train[index], cmap='gray')
plt.show()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def train_model(filters, kernel_size):
	model = Sequential()
	model.add(Input(shape=(28, 28, 1)))
	model.add(Conv2D(filters, kernel_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam',
				  metrics=['accuracy'])
	start = perf_counter()
	model.fit(X_train, y_train, validation_data=(X_test, y_test),
			  epochs=5, batch_size=200, verbose=0)
	training_seconds = perf_counter() - start
	scores = model.evaluate(X_test, y_test, verbose=0)
	return training_seconds, scores[0], scores[1]


base_time, base_loss, base_acc = train_model(32, (5, 5))
filter_time, filter_loss, filter_acc = train_model(64, (5, 5))
kernel_time, kernel_loss, kernel_acc = train_model(32, (3, 3))

print(f'Base model: filters=32, kernel_size=(5, 5), time={base_time:.2f}s, loss={base_loss:.4f}, accuracy={base_acc:.4f}')
print(f'64 filters, kernel=(5, 5): time={filter_time:.2f}s, loss={filter_loss:.4f}, accuracy={filter_acc:.4f}')
print(f'32 filters, kernel=(3, 3): time={kernel_time:.2f}s, loss={kernel_loss:.4f}, accuracy={kernel_acc:.4f}')
