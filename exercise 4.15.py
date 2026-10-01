# Exercise 4.15, based on Example 4.17 filter visualization
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()

# Layer 2 is the second convolutional layer, block1_conv2.
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print('Layer:', n, model.layers[n].name)
print('Input feature channels:', s[2])
print('Filter size:', s[0], s[1])
print('Total number of filters:', s[3])

# Normalize filter values to 0-1 so they can be visualized.
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
	image_filter = filters[:, :, :, i]
	for channel in range(s[2]):
		ax = pyplot.subplot(n_filters, s[2], ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(image_filter[:, :, channel], cmap='gray')
		ix += 1
pyplot.show()

# This layer combines lower-level edge/color responses into more complex
# local patterns than block1_conv1.
