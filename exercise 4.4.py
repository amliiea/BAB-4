#Example 4.4 LeNet-5 Keras: compare convolution filter counts
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers


def build_lenet(first_layer_filters):
	model = Sequential()
	model.add(Conv2D(filters=first_layer_filters, kernel_size=(3, 3),
					 activation='relu', input_shape=(32, 32, 1)))
	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=120, activation='relu'))
	model.add(Dense(units=84, activation='relu'))
	model.add(Dense(units=10, activation='softmax'))
	return model


original_model = build_lenet(6)
modified_model = build_lenet(12)
print('Example 4.4 parameters:', original_model.count_params())
print('Modified parameters (12 filters):', modified_model.count_params())
modified_model.summary()
