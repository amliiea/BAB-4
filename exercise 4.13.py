# Example 4.15: compare all available image-classifier models
import time

import cv2
import numpy as np
from tensorflow.keras.applications import (
	DenseNet201,
	InceptionV3,
	MobileNetV2,
	ResNet50,
	VGG16,
)
from tensorflow.keras.applications import densenet, inception_v3, mobilenet_v2
from tensorflow.keras.applications import resnet50, vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model_specs = [
	('vgg16', 224, VGG16, vgg16.preprocess_input),
	('resnet50', 224, ResNet50, resnet50.preprocess_input),
	('mobilenetv2', 224, MobileNetV2, mobilenet_v2.preprocess_input),
	('densenet201', 224, DenseNet201, densenet.preprocess_input),
	('inceptionv3', 299, InceptionV3, inception_v3.preprocess_input),
]
models = []
for model_name, image_size, model_factory, preprocess_input in model_specs:
	model = model_factory(weights='imagenet')
	models.append((model_name, image_size, model, preprocess_input))
	print(f'Loaded {model_name}')

camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Webcam could not be opened.')

inference_times = {name: [] for name, _, _, _ in models}
try:
	while camera.isOpened():
		ok, cam_frame = camera.read()
		if not ok:
			break
		for index, (model_name, image_size, model, preprocess_input) in enumerate(models):
			frame = cv2.resize(cam_frame, (image_size, image_size))
			image_batch = np.expand_dims(np.asarray(frame), 0)
			image_batch = preprocess_input(image_batch)
			start = time.perf_counter()
			predictions = model.predict(image_batch, verbose=0)
			inference_times[model_name].append(time.perf_counter() - start)
			label = decode_predictions(predictions, top=1)[0][0]
			if index < 4:
				cv2.putText(
					cam_frame,
					f'{model_name}: {label[1]}, {label[2]:.2f}',
					(10, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, (0, 255, 0), 2)
		cv2.imshow('Image classifier model comparison', cam_frame)
		if cv2.waitKey(30) == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()

for model_name, times in inference_times.items():
	if times:
		print(f'{model_name}: mean inference={np.mean(times) * 1000:.2f} ms, '
			  f'frames={len(times)}')
