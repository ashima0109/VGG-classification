# load the libraries

# import the VGG model from the module vgg.py
from vgg import *

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard,ReduceLROnPlateau
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import copy
import sys
from loguru import logger


logger.debug('All modules imported')

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train_model():
    # current working directory from which main.py is located
    cur_dir = os.getcwd()

    # the data is located in this data_dir
    data_dir = os.path.join(cur_dir, 'Dataset')

    # the output model and the graph is saved in this 'output_dir'
    output_dir = os.path.join(cur_dir, 'Output')

    logger.debug("[INFO] loading images...")

    data = []
    labels = []

    # grab the image paths and shuffle them
    # for reproducible results, use a constant seed for shuffling
    imagePaths = sorted(list(paths.list_images(data_dir) ))
    random.seed(2)
    random.shuffle(imagePaths)

    # loop over the input image paths
    for imagePath in imagePaths:

        # load the image
        image = cv2.imread(imagePath)

        # resize it
        image = cv2.resize(image, (64, 64))

        # append to the data list
        data.append(image)

        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    logger.debug('[INFO] data loaded complete...')



    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Binarize labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # save the encoder to output directory
    with open(os.path.join(output_dir,'labels'), 'wb') as f:
        pickle.dump(lb, f)

    # Randomly split the data into test and train sets (15% test and 85% train)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15, random_state=42)



    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=45, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    # initialize our VGG-like Convolutional Neural Network
    model = VGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))



    # initialize our initial learning rate, # of epochs to train for,and batch size
    INIT_LR = 0.0005
    EPOCHS = 150
    BS = 64

    # Checkpoints between the training steps
    model_checkpoint = ModelCheckpoint(filepath='VGG_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=20)
    # Termination of training if the loss become Nan
    terminate_on_nan = TerminateOnNaN()

    # For watching the live loss, accuracy and graphs using tensorboard
    t_board = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          batch_size=32,
                          write_graph=True,
                          write_grads=False,
                          write_images=False,
                          embeddings_freq=0,
                          update_freq='epoch')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=10, min_lr=0.0001)

    callbacks = [model_checkpoint, t_board, terminate_on_nan, reduce_lr]

    # initialize the model and optimizers
    opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, amsgrad=False)


    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    logger.debug('Training the network...')
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS,callbacks=callbacks)


    # Save the model locally for use later
    model_path = os.path.join(output_dir,  'path_to_my_VGG_model.h5')
    model.save(model_path)

    # evaluating the model
    logger.debug('Making predictions and evaluating the trained model.')
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy plots
    N = np.arange(0, EPOCHS)
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training/Validation Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir ,'vggnet_plot.png'))
