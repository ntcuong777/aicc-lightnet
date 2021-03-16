# Import darknet Changing the workdir is necessary to import the darknet's compile file
import os
WORKDIR = os.getcwd()
TEMPDIR = os.path.join(os.getcwd(), 'module/detector/detector_api/darknet')
os.chdir(TEMPDIR)
from module.detector.detector_api.darknet import darknet 
os.chdir(WORKDIR)
# Why is it so dark in here?

from module.detector.detector_model import DetectorModel
from module.detector.object.detection import Detection
import random
import numpy as np
import cv2 as cv

USE_DARKNET_RESIZE = False

DARKNET_MODELS_PARAMS = {
	"yolov4": {"cfg": "cfg/yolov4-aicc.cfg", 
			"weights": "weights/yolov4-aicc.weights"},
	"yolov4-csp": {"cfg": "cfg/yolov4-csp-aicc.cfg", 
			"weights": "weights/yolov4-csp-aicc.weights"},
	"yolov4x-mish": {"cfg": "cfg/yolov4x-mish-aicc.cfg", 
			"weights": "weights/yolov4x-mish-aicc.weights"},
	"yolov4-tiny": {"cfg": "cfg/yolov4-tiny-aicc.cfg", 
			"weights": "weights/yolov4-tiny-aicc.weights"},
	"yolov4-tiny-3l": {"cfg": "cfg/yolov4-tiny-3l-aicc.cfg", 
			"weights": "weights/yolov4-tiny-3l-aicc.weights"}
}

class DarknetDetector(DetectorModel):
	"""
	Class to make use of YOLOv4x-mish and YOLOv4-CSP detector model
	from compiled `darknet` library.

	It is fast, efficient, and hardcore.

	Attributes:
		network: `darknet`-based neural network model
		class_names: list of string labels
	"""
	def __init__(self, model_name, batch_size=2, data_file='aicc.data'):
		"""
			Params:
				model_name: either `yolov4`, `yolov4-csp`, `yolov4x-mish`, `yolov4-tiny`, or `yolov4-tiny-3l`
				data_file: path/to/data_file.data, the same format as coco.data in <darknet repo>/cfg/coco.data
		"""
		assert model_name in ["yolov4", "yolov4-csp", "yolov4x-mish", "yolov4-tiny", "yolov4-tiny-3l"],\
                "Error in DarknetDetector.__init__(model_name), `model_name` should be " + \
                "`yolov4`, `yolov4-csp`, `yolov4x-mish`, `yolov4-tiny`, or `yolov4-tiny-3l`."

		# Assembling config paths
		base_path 	 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'detector_api/darknet')
		config_path  = os.path.join(base_path, DARKNET_MODELS_PARAMS[model_name]["cfg"])
		weights_path = os.path.join(base_path, DARKNET_MODELS_PARAMS[model_name]["weights"])
		data_file	 = os.path.join(base_path, data_file)

		assert os.path.isfile(config_path), \
				"{0} does not exists".format(config_path)
		assert os.path.isfile(weights_path), \
				"{0} does not exists".format(weights_path)

		self.nms = True
		self.batch_size = batch_size
		
		# Load the network
		os.chdir(TEMPDIR)
		self.network, self.class_names, _ = darknet.load_network(
			config_path,
			data_file,
			weights_path,
			batch_size=self.batch_size
		)
		os.chdir(WORKDIR)


	def convert_to_vehicle(self, predictions):
		""" Convert the normalized detections and get `tlwh` bbox detections,
			Return the list of Vehicle instances.
		
		Args:
			predictions: a list, format = tuple( label: str, conf: float, bbox: tuple(xmin, ymin, w, h) )

		"""
		# Detection List
		detections = []
		for i in range(len(predictions)):
			bbox = np.array(predictions[i][2])
			bbox[0] -= bbox[2]/2
			bbox[1] -= bbox[3]/2
			score = predictions[i][1]
			actual_class = predictions[i][0]
			detections.append(Detection(bbox, score, actual_class))

		return detections


	def yolo_detect_image(self, image, min_confidence=.5, nms_max_overlap=.45):
		"""
			Returns a list of detections with their bbox.
			This part of code is borrowed and modified from darknet.py from the source repo.

			Params:
				image: darknet-based image.
		"""
		hier_thresh = .5 # Darknet get_network_boxes requires this for other type of layers
		pnum = darknet.pointer(darknet.c_int(0))
		darknet.predict_image(self.network, image) # Forward pass and save result at the top of network
		detections = darknet.get_network_boxes(self.network, image.w, image.h,
									0.5, hier_thresh, None, 0, pnum, 0) # retrieve detections
		num = pnum[0]
		if self.nms:
			darknet.do_nms_sort(detections, num, len(self.class_names), nms_max_overlap)
		
		# `predictions`: a list, format = tuple( label: str, conf: float, bbox: tuple(xmin, ymin, w, h) )
		predictions = darknet.remove_negatives(detections, self.class_names, num)
		darknet.free_detections(detections, num)
		return predictions


	def detect_objects(self, frame):
		# Unfortunately, Darknet does not integrated with numpy
		# so we cannot use numpy.ndarray images to pass into
		# the network. However, fortunately, we can convert
		# numpy images to darknet-based images with the following
		# lines of code. This should be intuitive enough though :)
		# We get the height and width to put into the network
		# and initialize an darknet-based image with the corresponding
		# height and width.
		height = None
		width = None
		if not USE_DARKNET_RESIZE:
			width = darknet.network_width(self.network)
			height = darknet.network_height(self.network)
		else:
			height = frame.shape[0]
			width = frame.shape[1]

		darknet_image = darknet.make_image(width, height, 3)

		# We need to convert opencv's BGR format to RGB format
		image_input = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

		if not USE_DARKNET_RESIZE: # Which resize is faster? CV Python or C-based Darknet?
			# Resize image to feed into the network
			# Larger input image scale is better according to YOLOv4 author
			image_input = cv.resize(image_input, (width, height), interpolation=cv.INTER_LINEAR)

		# This is where we actually handle the conversion from numpy images to darknet-based images
		darknet.copy_image_from_bytes(darknet_image, image_input.tobytes())

		# Conf suppression and nms suppression is automatically handled in using Darknet
		predictions = self.yolo_detect_image(image=darknet_image, min_confidence=.5, nms_max_overlap=.45)
		detections = self.convert_to_vehicle(predictions)

		darknet.free_image(darknet_image) # C needs to free pointers

		return detections


	def _darknet_prepare_batch(self, images, channels=3):
		"""
			This method prepare a batch of images in Darknet-compatible type
			for inference within the Darknet library
		"""

		# Unfortunately, Darknet does not integrated with numpy
		# so we cannot use numpy.ndarray images to pass into
		# the network. However, fortunately, we can convert
		# numpy images to darknet-based images with the following
		# lines of code. This should be intuitive enough though :)
		# We get the height and width to put into the network
		# and initialize an darknet-based image with the corresponding
		# height and width.
		height = None
		width = None
		if not USE_DARKNET_RESIZE:
			width = darknet.network_width(self.network)
			height = darknet.network_height(self.network)
		else:
			height = images.shape[1]
			width = images.shape[2]

		batch_array = []
		for image in images[:, ...]:
			image_input = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			if not USE_DARKNET_RESIZE: # Which resize is faster? CV Python or C-based Darknet?
				# Resize image to feed into the network
				# Larger input image scale is better according to YOLOv4 author
				image_input = cv.resize(image_input, (width, height), interpolation=cv.INTER_LINEAR)
			batch_array.append(image_input)

		batch_array = np.concatenate(batch_array, axis=0)
		darknet_images = darknet.make_image_batch(self.batch_size, width, height, channels)
		darknet.copy_image_batch_from_bytes(darknet_images, batch_array.tobytes())
		# return darknet.IMAGE(width, height, channels, darknet_images)
		# darknet_images = darknet.IMAGE(width, height, channels, darknet_images)
		return darknet_images


	def yolo_batch_detection(self, darknet_images, img_height, img_width, thresh=0.25, hier_thresh=.5, max_nms_overlap=.45):
		batch_detections = darknet.network_predict_batch_modified(self.network, darknet_images, self.batch_size, img_width,
                                                     img_height, thresh, hier_thresh, None, 0, 0)
		
		batch_predictions = []
		for idx in range(self.batch_size):
			num = batch_detections[idx].num
			detections = batch_detections[idx].dets
			if self.nms:
				darknet.do_nms_obj(detections, num, len(self.class_names), max_nms_overlap)
			predictions = darknet.remove_negatives(detections, self.class_names, num)
			batch_predictions.append(predictions)
		darknet.free_batch_detections(batch_detections, self.batch_size)
		return batch_predictions


	def detect_objects_by_batch(self, batch):
		"""
			Do a batch prediction.

			Input:
				batch: batch of images with prespecified batch_size at initialization time (in self.batch_size)
			Output:
				batch_detections: a list of each corresponding image's detections.
		"""
		darknet_images = self._darknet_prepare_batch(batch)
		batch_predictions = self.yolo_batch_detection(darknet_images, batch.shape[1], batch.shape[2])
		darknet.free_image_batch(darknet_images) # Free memory used to store the image batch

		batch_detections = []
		for predictions in batch_predictions:
			detections = self.convert_to_vehicle(predictions)
			batch_detections.append(detections)

		return batch_detections