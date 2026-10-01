import time

import cv2
import numpy as np
from tensorflow.keras.applications import densenet, resnet50
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

image_size = 224
models = [
	('ResNet50', resnet50.ResNet50(weights='imagenet'),
	 resnet50.preprocess_input),
	('DenseNet121', densenet.DenseNet121(weights='imagenet'),
	 densenet.preprocess_input),
]
camera = cv2.VideoCapture(0)
if not camera.isOpened():
	raise RuntimeError('Webcam could not be opened.')

inference_times = {name: [] for name, _, _ in models}
try:
	while camera.isOpened():
		ok, cam_frame = camera.read()
		if not ok:
			break
		for index, (model_name, model, preprocess) in enumerate(models):
			frame = cv2.resize(cam_frame, (image_size, image_size))
			image_batch = np.expand_dims(img_to_array(frame), axis=0)
			processed_image = preprocess(image_batch.copy())
			start = time.perf_counter()
			predictions = model.predict(processed_image, verbose=0)
			inference_times[model_name].append(time.perf_counter() - start)
			label = decode_predictions(predictions, top=1)[0][0]
			cv2.putText(
				cam_frame,
				f'{model_name}: {label[1]}, {label[2]:.2f}',
				(10, 30 + 35 * index), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (255, 0, 0), 2)
		cv2.imshow('ResNet50 and DenseNet121 webcam comparison', cam_frame)
		if cv2.waitKey(30) == 27:
			break
finally:
	camera.release()
	cv2.destroyAllWindows()

for model_name, times in inference_times.items():
	if times:
		print(f'{model_name}: mean inference={np.mean(times) * 1000:.2f} ms, '
			  f'frames={len(times)}')
