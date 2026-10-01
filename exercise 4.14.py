# Example 4.15: choose a classifier with if-else statements
import cv2
import numpy as np
from tensorflow.keras.applications import (
	densenet,
	inception_v3,
	mobilenet_v2,
	resnet50,
	vgg16,
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model_name = 'vgg16'
if model_name == 'vgg16':
	model_factory = vgg16.VGG16
	preprocess_input = vgg16.preprocess_input
	image_size = 224
elif model_name == 'resnet50':
	model_factory = resnet50.ResNet50
	preprocess_input = resnet50.preprocess_input
	image_size = 224
elif model_name == 'mobilenetv2':
	model_factory = mobilenet_v2.MobileNetV2
	preprocess_input = mobilenet_v2.preprocess_input
	image_size = 224
elif model_name == 'densenet201':
	model_factory = densenet.DenseNet201
	preprocess_input = densenet.preprocess_input
	image_size = 224
elif model_name == 'inceptionv3':
	model_factory = inception_v3.InceptionV3
	preprocess_input = inception_v3.preprocess_input
	image_size = 299
else:
	raise ValueError(f'Unsupported model: {model_name}')

model = model_factory(weights='imagenet')
model.summary()
camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Webcam could not be opened.')

try:
	while camera.isOpened():
		ok, cam_frame = camera.read()
		if not ok:
			break
		frame = cv2.resize(cam_frame, (image_size, image_size))
		image_batch = np.expand_dims(np.asarray(frame), 0)
		image_batch = preprocess_input(image_batch)
		predictions = model.predict(image_batch, verbose=0)
		label = decode_predictions(predictions, top=1)[0][0]
		cv2.putText(
			cam_frame, f'{model_name}: {label[1]}, {label[2]:.2f}',
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
		cv2.imshow('Classification', cam_frame)
		if cv2.waitKey(30) == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()
