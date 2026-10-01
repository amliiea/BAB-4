#Example 4.9 Simple CNN for the MNIST Dataset: add a second block
from time import perf_counter

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
for index in range(4):
	plt.subplot(141 + index)
	plt.imshow(X_train[index])
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


def train_model(add_block):
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	if add_block:
		model.add(Conv2D(32, (5, 5), activation='relu'))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam',
				  metrics=['accuracy'])
	start = perf_counter()
	model.fit(X_train, y_train, validation_data=(X_test, y_test),
			  epochs=10, batch_size=200, verbose=0)
	training_seconds = perf_counter() - start
	scores = model.evaluate(X_test, y_test, verbose=0)
	accuracy = scores[1]
	return training_seconds, accuracy, 100 - accuracy * 100


accuracies = {}
for add_block in (False, True):
	training_seconds, accuracy, error = train_model(add_block)
	accuracies[add_block] = accuracy
	print(f'additional convolution block={add_block}: '
		  f'training seconds={training_seconds:.2f}, '
		  f'accuracy={accuracy * 100:.2f}%, CNN Error={error:.2f}%')
print('Accuracy change with additional block: %.2f percentage points' %
	  ((accuracies[True] - accuracies[False]) * 100))
