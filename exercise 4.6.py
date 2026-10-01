#Example 4.7 custom CNN: add Conv2D, MaxPooling2D, and Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def build_model(add_block):
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
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam',
				  metrics=['accuracy'])
	return model


original_model = build_model(False)
modified_model = build_model(True)
print('Example 4.7 parameters:', original_model.count_params())
print('Modified parameters:', modified_model.count_params())
print(modified_model.summary())
