# Example 4.17 CNN Visualize Filters.ipynb
#Modified from:
#https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import expand_dims

# load vgg model
model = VGG19()
# summarize the model
model.summary()

# display all layers and their layer numbers
n = 0
for layer in model.layers:
	print(n, layer.name)
	n += 1

# show filter and bias shapes for the convolutional layers
n = 0
for layer in model.layers:
	if 'conv' in layer.name:
		filters, biases = layer.get_weights()
		print(n, layer.name, filters.shape, biases.shape)
	n += 1

# visualize the first four filters in the first convolutional layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(s[0]):
		ax = pyplot.subplot(n_filters, s[0], ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(f[j, :, :], cmap='gray')
		ix += 1
pyplot.show()


def plot_feature_maps(feature_maps):
	# plot all feature maps
	col = 8
	row = int(feature_maps.shape[3] / col)
	ix = 1
	plt.figure(figsize=(20, 20))
	for _ in range(row):
		for _ in range(col):
			ax = plt.subplot(row, col, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
			ix += 1
	plt.show()


# load the model and select a hidden layer to visualize
model = VGG19()
n = 1
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# load and prepare the image
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
from keras.applications.vgg19 import preprocess_input
img = preprocess_input(img)

# get and display the feature maps
feature_maps = model.predict(img, verbose=0)
print("Feature maps: ", feature_maps.shape)
plot_feature_maps(feature_maps)
