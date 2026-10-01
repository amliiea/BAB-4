# Exercise 4.16, based on Example 4.17 filter visualization
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print('Layer:', n, model.layers[n].name)
print('Color channels:', s[2])
print('Filter size:', s[0], s[1])
print('Total number of filters:', s[3])

# Normalize filter values to 0-1 for visualization.
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = s[3], 1
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

# All 64 filters are shown, with one panel for each of the three input channels.
