
# import the necessary packages
import numpy as np
import argparse
import cv2
import os

main_dir = os.getcwd()
output_dir = os.path.join(main_dir, 'output')

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument("-train", action='store_true',
	help="trains the model")
parser.add_argument("-predict", "--predict", required=False,
	help="path to file")

args = vars(parser.parse_args())

# for training
if args['train']:
	import train
	train.train_model()

# for prediction
if args['predict']:
	image_path = args['predict']

	# check for a valid image path
	if os.path.isfile(image_path):
		import predict
		predict.make_prediction(image_path, output_dir)
	else:
		print('Please provide a valid image path')
