***


## Directory map

If you clone the Git repo, you will end up with this structure below:

```
.
├── Dataset
│   ├── black_dress [450 entries]
│   ├── black_pants [539 entries]
│   ├── blue_dress [502 entries]
│   ├── blue_pants [512 entries]
│   ├── green_shoes [455 entries]
│   ├── red_dress [527 entries]
│   ├── red_shoes [501 entries]
│   └── white_dress [506 entries]
├── Output
├── test_examples_labels
├── logs
├── README.md
├── main.py
├── predict.py
├── train.ipynb
└── vgg.py
```

The different categories of the data are stored under the folder **Dataset**. Each category has ~450-550 entries. The trained model and the loss-accuracy plots would be saved in **Output** folder. The **logs** folder is for storing the checkpoints for monitoring the progress of loss and accuracies live on the tensorboard while the model is being trained. The folder **test\_examples\_labels** has some examples of images not in the dataset for prediction. A complete guide for running the code is available at my [blog](https://medium.com/@ashima0109/image-classification-with-vgg-convolutional-neural-network-using-keras-for-beginners-61767950c5dd).

###Requirements

* keras == 2.2.4
* scikit-learn == 0.21.2
* pandas == 0.24.2
* numpy == 1.16.4
* opencv == 3.4.2
* matplotlib == 3.1.0


###Run Instructions
To **train** the model, run the following command through the command line: 
``` 
python3 main.py -train 
```

To **predict** the class label of an unseen image, run the command
``` 
python3 main.py -predict {path_to_the_image}
```

The output would be displayed on the terminal and an image with name 'output.jpg' would be saved with the label and the confidence value for visualization.







