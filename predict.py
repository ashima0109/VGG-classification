
# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import logging, os

# disable the warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# function to process the image similar to the training process
def process_image(image_path):

	# load the input image and resize it to the target spatial dimensions
	image = cv2.imread(image_path)
	width, height = 64,64

	image = cv2.resize(image, (width, height))

	# image to array
	b = img_to_array(image)

	# normalize the image
	processed_image = np.array(b, dtype="float") / 255.0

	return processed_image


def make_prediction(image_path, model_dir):

	# process the image
	image = process_image(image_path)

	# read the image for the output
	output = cv2.imread(image_path)

	# load the model and label binarizer from the directory
	print("[INFO] loading model and label binarizer...")

	# relative paths to the model and labels
	model_path = os.path.join(model_dir, 'trained_VGG_model.h5')
	label_file_path = os.path.join(model_dir, 'labels')

	# load the model and the label encoder
	model = load_model(model_path)
	lb = pickle.loads(open(label_file_path, "rb").read())

	# make a prediction on the image
	image = np.expand_dims(image, axis = 0)
	pred_result = model.predict(image)

	# extract the class label which has the highest corresponding probability
	i = pred_result.argmax(axis=1)[0]
	label = lb.classes_[i]

	# draw the class label + probability on the output image
	text = "{}: {:.2f}%".format(label, pred_result[0][i] * 100)
	cv2.putText(output, text, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

	# display the result on the screen
	print("Predicted label {}: {:.2f}%".format(label, pred_result[0][i] * 100))

	# save the output image with label
	cv2.imwrite('output.jpg', output)
