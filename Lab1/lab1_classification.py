import os
import numpy as np
import torch
import torchvision
import tensorflow as tf                                     # NOTE: This is just for utils. Do not use tensorflow in your code.
from tensorflow.keras.utils import to_categorical           # NOTE: This is just for utils. Do not use tensorflow in your code.
import torch.nn.functional as F
import torch.optim as optim
import random
from torch import nn

NEURONS_PER_LAYER_NUMPY = 32
# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)
torch.manual_seed(1618)

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_ann"
#ALGORITHM = "pytorch_ann"
ALGORITHM = "pytorch_cnn"

#DATASET = "mnist_d"             # Handwritten digits.
#DATASET = "mnist_f"            # Scans of types of clothes.
#DATASET = "cifar_10"           # Color images (10 classes).
DATASET = "cifar_100_f"        # Color images (100 classes).
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
        return 1 / (1 + np.exp(-x))
    # Activation prime function.
    def _sigmoidDerivative(self, x):
        return x * (1 - x)
    def _reLu(self, x):
        return np.maximum(0, x)
    def _reLuDerivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def activation(self, xVals, activation):
        if activation == "sigmoid":
            return self._sigmoid(xVals)
        elif activation == "reLu":
            return self._reLu(xVals)
    def activationDerivative(self, xVals, activation):
        if activation == "sigmoid":
            return self._sigmoidDerivative(xVals)
        elif activation == "reLu":
            return self._reLuDerivative(xVals)
    # Batch generator for mini-batches. Not randomized.
    def _batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
    # Training with backpropagation.
    def train(self, xVals, yVals, activation="sigmoid",epochs = 100, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        for i in range(epochs):
            if minibatches:
                for x_batch, y_batch in zip(self._batchGenerator(xVals, mbs), self._batchGenerator(yVals, mbs)):
                    layer1, layer2 = self._forward(x_batch, activation)
                    L2e = -(layer2 - y_batch)
                    L2d = L2e*self.activationDerivative(layer2, activation)
                    L1e = L2d.dot(self.W2.T)
                    L1d = L1e*self.activationDerivative(layer1, activation)
                    L1a = (x_batch.T.dot(L1d))*self.lr
                    L2a = (layer1.T.dot(L2d))*self.lr
                    self.W1 += L1a
                    self.W2 += L2a
            else:
                layer1, layer2 = self._forward(xVals, activation)
                L2e = -(layer2 - y_batch)
                L2d = L2e*self.activationDerivative(layer2, activation)
                L1e = L2d.dot(self.W2.T)
                L1d = L1e*self.activationDerivative(layer1, activation)
                L1a = (xVals.T.dot(L1d))*self.lr
                L2a = (layer1.T.dot(L2d))*self.lr
                self.W1 += L1a
                self.W2 += L2a

    # Forward pass.
    def _forward(self, input, activation="sigmoid"):
        layer1 = self.activation(np.dot(input, self.W1), activation)
        layer2 = self.activation(np.dot(layer1, self.W2), activation)
        return layer1, layer2

    # Predict.
    def __call__(self, xVals):
        _, layer2 = self._forward(xVals)
        output = np.zeros_like(layer2)
        output[np.arange(len(layer2)), layer2.argmax(1)] = 1
        return output


#==========================<Pytorch Neural Net>=================================

'''
    A neural net built around nn.Linear.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_ANN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Pytorch_ANN, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = [512, 256, 128]
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(nn.Linear(self.inputSize, self.hiddenSize[0]),
                                nn.ReLU(),
                                nn.Linear(self.hiddenSize[0], self.hiddenSize[1]),
                                nn.ReLU(),
                                nn.Linear(self.hiddenSize[1], self.hiddenSize[2]),
                                nn.ReLU(),
                                nn.Linear(self.hiddenSize[2], outputSize)).to(self.device)
        
    def generate_batch(self, x, y, batch_size):
        for i in range(0, len(x), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]      
    def forward(self, x):
        tensor_x = torch.from_numpy(x).float()
        tensor_x = tensor_x.to(self.device)
        return self.model(tensor_x)
        
    def train(self, x, y, epochs = 500):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        tensor_y = torch.from_numpy(y).float()
        tensor_y = tensor_y.to(self.device)
        for i in range(epochs):
            for x_batch, y_batch in self.generate_batch(x, tensor_y, 500):
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
    def __call__(self, x):
        output = self.forward(x)
        # covert tensor to numpy array
        output = output.cpu().detach().numpy()
        result = np.zeros_like(output)
        result[np.arange(len(output)), output.argmax(1)] = 1
        return result



#==========================<Pytorch Convolutional Net>==========================

'''
    A neural net built around nn.Conv2d with one nn.Linear as the output layer.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_CNN(nn.Module):
    def __init__(self, input_W, input_H, channels, outputSize):
        conv_size = 4
        if input_W == 28: conv_size = 3
        
        super(Pytorch_CNN, self).__init__()
        covnv_filter = [32, 64, 128, 128, 256, 256]
        conv_kernel = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(nn.Conv2d(channels, covnv_filter[0], conv_kernel, padding=1),
                                nn.BatchNorm2d(covnv_filter[0]),
                                #########################################
                                nn.Conv2d(covnv_filter[0], covnv_filter[1], conv_kernel, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(covnv_filter[1]),
                                #########################################
                                nn.Conv2d(covnv_filter[1], covnv_filter[2], conv_kernel, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(covnv_filter[2]),
                                #########################################
                                nn.Conv2d(covnv_filter[2], covnv_filter[3], conv_kernel, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(covnv_filter[3]),
                                #########################################
                                nn.Conv2d(covnv_filter[3], covnv_filter[4], conv_kernel, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.BatchNorm2d(covnv_filter[4]),
                                #########################################
                                nn.Conv2d(covnv_filter[4], covnv_filter[5], conv_kernel, padding=1),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.BatchNorm2d(covnv_filter[5]),
                                #########################################
                                nn.Flatten(),  
                                nn.Linear(256*conv_size*conv_size, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, outputSize)).to(self.device)
    def generate_batch(self, x, y, batch_size):
        for i in range(0, int(len(x)), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]
    def train(self, x, y, xTest, yTest, epochs = 50):
        tensor_x = torch.from_numpy(x).float().to(self.device)
        tensor_y = torch.from_numpy(y).float().to(self.device)
        
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            correct = 0
            for x_batch, y_batch in self.generate_batch(tensor_x, tensor_y, 1024):
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                # calculate training accuracy
                output = output.cpu().detach().numpy()
                pred = np.zeros_like(output)
                pred[np.arange(len(output)), output.argmax(1)] = 1
                correct += (pred == y_batch.cpu().detach().numpy()).all(axis=-1).sum()
            # calculate validation accuracy
            x_test = torch.from_numpy(xTest).float().to(self.device)
            valid_output = self.model(x_test)
            valid_output = valid_output.cpu().detach().numpy()
            valid_pred = np.zeros_like(valid_output)
            valid_pred[np.arange(len(valid_output)), valid_output.argmax(1)] = 1
            valid_acc = (valid_pred == yTest).all(axis=-1).sum() 

            print("Epoch:", epoch+1 , "| Loss:", round(loss.item(),5), "| Training Accuracy:",round(100. * correct / len(x),2), "%", "| Validation Accuracy:", round(100. * valid_acc /len(xTest),2), "%")
    def __call__(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        output = self.model(x)
        # covert tensor to numpy array
        output = output.cpu().detach().numpy()
        result = np.zeros_like(output)
        result[np.arange(len(output)), output.argmax(1)] = 1
        return result



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
    ((xTrain, yTrain), (xTest, yTest)) = raw  #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = xTrain/255.0 
    xTest = xTest/255.0
    if (ALGORITHM != "pytorch_cnn"):
        xTrain = xTrain.reshape(xTrain.shape[0], imgW*imgH)
        xTest = xTest.reshape(xTest.shape[0], imgW*imgH)
    else:
        xTrain = xTrain.reshape(-1, imgC, imgW, imgH)
        xTest = xTest.reshape(-1, imgC, imgW, imgH)
    yTrainP = to_categorical(yTrain, numClasses)
    yTestP = to_categorical(yTest, numClasses)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data, test, numClasses, imgW, imgH, imgC):
    xTrain, yTrain = data
    xTest, yTest = test
    if ALGORITHM == "guesser":
        return guesserClassifier   # Guesser has no training, as it is just guessing.
    elif ALGORITHM == "custom_ann":
        print("Building and training Custom_ANN.")
        c_ann = Custom_ANN(inputSize=imgW*imgH*imgC, outputSize=numClasses, neuronsPerLayer=NEURONS_PER_LAYER_NUMPY)
        c_ann.train(xTrain, yTrain)
        return c_ann
    elif ALGORITHM == "pytorch_ann":
        print("Building and training Pytorch_ANN.")
        p_ann = Pytorch_ANN(inputSize=imgW*imgH*imgC, outputSize=numClasses)
        p_ann.train(xTrain, yTrain)
        return p_ann
    elif ALGORITHM == "pytorch_cnn":
        print("Building and training Pytorch_CNN.")
        p_cnn = Pytorch_CNN(imgW, imgH, imgC, outputSize=numClasses)
        p_cnn.train(xTrain, yTrain, xTest, yTest)
        return p_cnn
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
    model = trainModel(data[0], data[1], nc, w, h, ch)
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()