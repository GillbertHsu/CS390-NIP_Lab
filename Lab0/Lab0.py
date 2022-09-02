import os
import numpy as np
import torch
import torchvision
import tensorflow as tf                                     # NOTE: This is just for utils. Do not use tensorflow in your code.
from tensorflow.keras.utils import to_categorical           # NOTE: This is just for utils. Do not use tensorflow in your code.
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)
torch.manual_seed(1618)

# Use these to set the algorithm to use.
ALGORITHM = "guesser"
#ALGORITHM = "custom_ann"
#ALGORITHM = "pytorch_ann"
#ALGORITHM = "pytorch_cnn"

DATASET = "mnist_d"             # Handwritten digits.
#DATASET = "mnist_f"            # Scans of types of clothes.
#DATASET = "cifar_10"           # Color images (10 classes).
#DATASET = "cifar_100_f"        # Color images (100 classes).
#DATASET = "cifar_100_c"        # Color images (20 classes).


'''
    In this lab, you will build a classifier system for several datasets and using several algorithms.
    Select your algorithm by setting the ALGORITHM constant.
    Select your dataset by setting the DATASET constant.

    Start by testing out the datasets with the guesser (already implemented).
    After that, I suggest starting with custom_ann on mnist_d.
    Once your custom_ann works on all datasets, try to implement pytorch_ann and pytorch_cnn.
    You must also add a confusion matrix and F1 score to evalResults.

    Feel free to change any part of this skeleton code.
'''



#==========================<Custom Neural Net>==================================

'''
    A neural net with 2 layers (one hidden, one output).
    Implement only with numpy, NO PYTORCH.
    A minibatch generator is already given.
'''

class Custom_ANN():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Activation prime function.
    def _sigmoidDerivative(self, x):
        return x * (1 - x)

    # Batch generator for mini-batches. Not randomized.
    def _batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100, minibatches = True, mbs = 100):
        pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def _forward(self, input):
        layer1 = self._sigmoid(np.dot(input, self.W1))
        layer2 = self._sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def __call__(self, xVals):
        _, layer2 = self._forward(xVals)
        return layer2


#==========================<Pytorch Neural Net>=================================

'''
    A neural net built around nn.Linear.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_ANN(nn.Module):
    def __init__(self):
        super(Pytorch_ANN, self).__init__()

    def forward(self, x):
        raise NotImplementedError()



#==========================<Pytorch Convolutional Net>==========================

'''
    A neural net built around nn.Conv2d with one nn.Linear as the output layer.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_CNN(nn.Module):
    def __init__(self):
        super(Pytorch_CNN, self).__init__()

    def forward(self, x):
        raise NotImplementedError()



#===============================<Random Classifier>=============================

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        numClasses = 10
        imgW, imgH, imgC = (28, 28, 1)
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        numClasses = 10
        imgW, imgH, imgC = (28, 28, 1)
    elif DATASET == "cifar_10":
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
        numClasses = 10
        imgW, imgH, imgC = (32, 32, 3)
    elif DATASET == "cifar_100_f":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="fine")
        numClasses = 100
        imgW, imgH, imgC = (32, 32, 3)
    elif DATASET == "cifar_100_c":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="coarse")
        numClasses = 20
        imgW, imgH, imgC = (32, 32, 3)
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return (((xTrain, yTrain), (xTest, yTest)), numClasses, imgW, imgH, imgC)



def preprocessData(raw, numClasses, imgW, imgH, imgC):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, numClasses)
    yTestP = to_categorical(yTest, numClasses)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data, numClasses, imgW, imgH, imgC):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return guesserClassifier   # Guesser has no training, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    return model(data)



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw, nc, w, h, ch = getRawData()
    data = preprocessData(raw, nc, w, h, ch)
    model = trainModel(data[0], nc, w, h, ch)
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()