# -*- coding: utf-8 -*-
"""nn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xYqQa9C28SYmPjZl7_8lQc5g5Ciqd-U5
"""

################################################################################
#
# LOGISTICS
#
#    NAME: Nikhil Darwin Bollepalli
#    ID:   NXB200019
#    
#
# FILE
#
#    nn.py
#
# DESCRIPTION
#
#    Grade = nn.py grade (max 80) + sw.py grade (max 20) + cnn.py grade (max 20)
#
#    This file is required; see above for grade calculation
#
#    This is the start of an exceedingly simple / lite / reduced functionality
#    PyTorch style xNN library written in Python and it's example use for MNIST
#    image classification
#
#    This code does not use PyTorch, TensorFlow or any other xNN library
#
#    NN specification:
#
#       ----------------------------   -------
#       Data loader                    Output
#       ----------------------------   -------
#       Data                           1x28x28
#       Division by 255.0              1x28x28
#
#       ----------------------------   -------
#       Network                        Output
#       ----------------------------   -------
#       Vectorization                  1x784
#       Vector matrix multiplication   1x100
#       Vector vector addition         1x100
#       ReLU                           1x100
#       Vector matrix multiplication   1x100
#       Vector vector addition         1x100
#       ReLU                           1x100
#       Vector matrix multiplication   1x10
#       Vector vector addition         1x10
#
#       ----------------------------   -------
#       Error                          Output
#       ----------------------------   -------
#       Softmax                        1x10
#       Cross entropy                  1
#
# INSTRUCTIONS
#
#    1. Complete all <TO DO: ...> code portions of this file
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics
#
#       
  # Epoch   0 Time     32.0 lr = 0.000100 avg loss = 1.114983 accuracy = 86.70
  # Epoch   1 Time     32.1 lr = 0.003400 avg loss = 0.226087 accuracy = 95.40
  # Epoch   2 Time     32.0 lr = 0.006700 avg loss = 0.130239 accuracy = 96.12
  # Epoch   3 Time     32.0 lr = 0.010000 avg loss = 0.109192 accuracy = 93.86
  # Epoch   4 Time     31.9 lr = 0.009699 avg loss = 0.073910 accuracy = 96.36
  # Epoch   5 Time     31.9 lr = 0.008831 avg loss = 0.052902 accuracy = 97.40
  # Epoch   6 Time     31.9 lr = 0.007502 avg loss = 0.036914 accuracy = 97.87
  # Epoch   7 Time     31.9 lr = 0.005872 avg loss = 0.022554 accuracy = 97.85
  # Epoch   8 Time     31.9 lr = 0.004138 avg loss = 0.013180 accuracy = 98.18
  # Epoch   9 Time     31.9 lr = 0.002507 avg loss = 0.008136 accuracy = 98.30
  # Epoch  10 Time     32.1 lr = 0.001179 avg loss = 0.005525 accuracy = 98.31
  # Epoch  11 Time     32.0 lr = 0.000311 avg loss = 0.004485 accuracy = 98.29
  # Epoch  12 Time     31.9 lr = 0.000010 avg loss = 0.004097 accuracy = 98.28
#
#    3. Submit nn.py via eLearning (no zip files, no Jupyter / iPython
#       notebooks, ...) with this comment block at the top and all code from
#       the IMPORT comment block to the end
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

import os.path
import urllib.request
import gzip
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_NORM              = np.float32(255.0)
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# model
MODEL_N0 = DATA_ROWS*DATA_COLS
MODEL_N1 = 100
MODEL_N2 = 100
MODEL_N3 = DATA_CLASSES

# training
TRAIN_LR_MAX          = 0.01
TRAIN_LR_INIT_SCALE   = 0.01
TRAIN_LR_FINAL_SCALE  = 0.001
TRAIN_LR_INIT_EPOCHS  = 3
TRAIN_LR_FINAL_EPOCHS = 10
TRAIN_NUM_EPOCHS      = TRAIN_LR_INIT_EPOCHS + TRAIN_LR_FINAL_EPOCHS
TRAIN_LR_INIT         = TRAIN_LR_MAX*TRAIN_LR_INIT_SCALE
TRAIN_LR_FINAL        = TRAIN_LR_MAX*TRAIN_LR_FINAL_SCALE

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

# x=np.random.standard_normal((10,3)).astype(np.float32)
# x0 = np.ones(3)
# w1 = np.random.standard_normal((3,5)).astype(np.float32)
# print(x0,x0.shape)
# print(w1,w1.shape)
# print()
# x1 = x0.dot(w1)
# print(x1) # x0*W
# print()

# w2 = np.zeros(5)
# print(w2) 
# x2 = x1+w2
# print(x2) # x0W + w2
# print(np.maximum(0,x2)) #eliminating -ve values
# print()

# print(w1.resize(15,1))

################################################################################
#
# DATA
#
################################################################################

# # download
# if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
#     urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
# if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
#     urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
# if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
#     urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
# if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
#     urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)


# download data
opener = urllib.request.URLopener()
opener.addheader('User-Agent', 'Mozilla/5.0')
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    opener.retrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    opener.retrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    opener.retrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    opener.retrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

# x0    = np.reshape(train_data[1, :, :, :], MODEL_N0)/DATA_NORM
# print(x0)

# train_data[1, :, :, :]

################################################################################
#
# DATA LOADER
#
################################################################################

# data loader class
class DataLoader:


    # save images x, labels y and data normalization factor x_norm
    def __init__(self, x, y, x_norm):
        # <TO DO: your code goes here>
        self.x = x
        self.y = y
        self.x_norm = x/DATA_NORM

    # return normalized image t and label t
    def get(self, t):
        # <TO DO: your code goes here>
        return np.reshape(self.x_norm[t, :, :, :], 784), self.y[t]

    # return the total number of images
    def num(self):
        # <TO DO: your code goes here>
        return self.x.shape[0]
# data loaders
data_loader_train = DataLoader(train_data, train_labels, DATA_NORM)
data_loader_test  = DataLoader(test_data,  test_labels,  DATA_NORM)

# x= data_loader_train.get(2)
# # print(x)
# data_loader_test.num()

# x = np.zeros(10).shape
# print(train_labels.shape)

################################################################################
#
# LAYERS
#
################################################################################

# vector matrix multiplication layer
class VectorMatrixMultiplication:

    # initialize input x, parameters h and parameter gradient de/dh
    def __init__(self, x_channels, y_channels):

        # initialisng input x
        self.x = np.zeros(x_channels, dtype=np.float32)
        
        # initialisng weights h (w1) randomly and bias w2 with zeros
        self.w1 = np.sqrt(2.0/(x_channels + y_channels), dtype=np.float32)*np.random.standard_normal((x_channels, y_channels)).astype(np.float32)
        # self.w2 = np.zeros(y_chany_channels, dtype=np.float32)
        
        # initialisng gradient de/dw
        self.dedw1 = np.zeros((x_channels, y_channels), dtype=np.float32)
        # self.dedw2 = np.zeros(y_channels, dtype=np.float32)

    # save the input x and return y = f(x, h)
    def forward(self, x):
        self.x = x
        return x.dot(self.w1)

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):

        # <TO DO: your code goes here>
        self.dedw1 = np.outer(self.x,dedy)
        # self.dedw1 =   ?
        # print(((self.w1).T).shape)
        # print("-----------")
        # print("dedy shape : ", dedy.shape)
        # print("self w1 : ", (self.w1).shape)
        # print("self w1.T : ", ((self.w1).T).shape)
        # print("------------")
        return dedy.dot((self.w1).T)

# vector vector addition layer
class VectorVectorAddition:

    # initialize parameters h and parameter gradient de/dh
    def __init__(self, x_channels):
        self.w2 = np.zeros(x_channels, dtype=np.float32)
        self.dedw2 = np.zeros(x_channels, dtype=np.float32)
    
    # return y = f(x, h)
    def forward(self, x):
        return x+self.w2

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        
        # <TO DO: your code goes here>
        self.dedw2 = dedy
        return dedy

# ReLU layer
class ReLU:

    # initialize input x
    def __init__(self, x_channels):
        self.x = x_channels

    # save the input x and return y = f(x, h)
    def forward(self, x):
        self.x = x
        return np.maximum(np.float32(0.0),x)

    # return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        # <TO DO: your code goes here>
        return np.minimum(np.float32(1.0), np.ceil(np.maximum(np.float32(0.0), self.x)))*dedy

# soft max cross entropy layer
class SoftMaxCrossEntropy:

    # initialize probability p and label
    def __init__(self, y_channels):
        # <TO DO: your code goes here>
        self.p = np.zeros(y_channels, dtype=np.float32)
        self.label = -1
    
    # save the label, compute and save the probability p and return e = f(label, y)
    def forward(self, label, y):  # expecting y to be np.exp(x8 )  x8 --> final output
        # <TO DO: your code goes here>
        self.label = label 
        self.p = np.exp(y)/np.sum(np.exp(y))
        loss = -np.log(self.p[label]) 
        # print("los :",loss)
        return loss

    # compute and return the input gradient de/dx from the saved probability and label; e is not used
    def backward(self, e):
        
        # <TO DO: your code goes here>
        self.p[label] = self.p[label] - np.float32(1.0) 
        return self.p  # equivalent of dedx8

################################################################################
#
# NETWORK
#
################################################################################

# network
class Network:

    # save the network description parameters and create all layers
    def __init__(self, rows, cols, n0, n1, n2, n3):
        # <TO DO: your code goes here>
        self.rows =rows
        self.cols =cols
        self.n0 =n0  #784
        self.n1 =n1  #100 
        self.n2 =n2  #100 
        self.n3 =n3  #10 
        self.vm1 = VectorMatrixMultiplication(self.n0,self.n1)
        self.vm2 = VectorMatrixMultiplication(self.n1,self.n2)
        self.vm3 = VectorMatrixMultiplication(self.n2,self.n3)

        self.va1= VectorVectorAddition(self.n1)
        self.va2= VectorVectorAddition(self.n2)
        self.va3= VectorVectorAddition(self.n3)

        self.r1 = ReLU(self.n1)
        self.r2 = ReLU(self.n2)
        # self.r3 = ReLU(self.n3)
        


    # connect layers forward functions together to map the input image to the network output
    # return the network output
    def forward(self, img):
        # <TO DO: your code goes here>
        # img = 28*28 & probably use vectormulti, vectvet addition & relu classes

        x0 = img 

        i1=self.r1.forward(self.va1.forward(self.vm1.forward(x0)))
        i2=self.r2.forward(self.va2.forward(self.vm2.forward(i1)))
        i3=self.va3.forward(self.vm3.forward(i2))

        return i3


    # connect layers backward functions together to map de/dy at the end of the network to de/dx at the beginning
    # note that inside the backward functions de/dh is computed for all parameters
    # optionally return de/dx (unused)
    def backward(self, dedy):
        # <TO DO: your code goes here>
        a1=self.va3.backward(dedy)
        # print(a1.shape)

        i1=self.r2.backward(self.vm3.backward(a1))
        # print(i1.shape)

        a2=self.va2.backward(i1)
        # print(a2.shape)

        i2=self.r1.backward(self.vm2.backward(a2))
        # print(i2.shape)

        a3=self.va1.backward(i2)
        # print(a3.shape)

        i3=self.vm1.backward(a3)
        # print(i3.shape)
        return i3 

    # update all layers with trainable parameters via h = h - lr * de/dh
    def update(self, lr):
        # <TO DO: your code goes here>
        self.vm1.w1 -= lr*(self.vm1.dedw1)
        self.va1.w2 -= lr*(self.va1.dedw2)

        self.vm2.w1 -= lr*(self.vm2.dedw1)
        self.va2.w2 -= lr*(self.va2.dedw2)

        self.vm3.w1 -= lr*(self.vm3.dedw1)
        self.va3.w2 -= lr*(self.va3.dedw2)




# network
network = Network(DATA_ROWS, DATA_COLS, MODEL_N0, MODEL_N1, MODEL_N2, MODEL_N3)

################################################################################
#
# ERROR
#
################################################################################

# error
error = SoftMaxCrossEntropy(MODEL_N3)

################################################################################
#
# UPDATE
#
################################################################################

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup
    if epoch < TRAIN_LR_INIT_EPOCHS:
        lr = (TRAIN_LR_MAX - TRAIN_LR_INIT)*(float(epoch)/TRAIN_LR_INIT_EPOCHS) + TRAIN_LR_INIT
    # 1/2 wave cosine decay
    else:
        lr = TRAIN_LR_FINAL + 0.5*(TRAIN_LR_MAX - TRAIN_LR_FINAL)*(1.0 + math.cos(((float(epoch) - TRAIN_LR_INIT_EPOCHS)/(TRAIN_LR_FINAL_EPOCHS - 1.0))*math.pi))

    return lr



################################################################################
#
# TRAIN
#
################################################################################

# initialize the epoch
start_epoch      = 0
start_time_epoch = time.time()

# initialize the display statistics
epochs   = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.int32)
avg_loss = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.float32)
accuracy = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.float32)

# cycle through the epochs
for epoch in range(start_epoch, TRAIN_NUM_EPOCHS):

    # set the learning rate
    lr = np.float32(lr_schedule(epoch))

    # initialize the epoch statistics
    training_loss   = 0.0
    testing_correct = 0

    # cycle through the training data
    for t in range(data_loader_train.num()):

        # data
        img, label = data_loader_train.get(t)

        # network forward pass, error forward pass, error backward pass and network backward pass
        y    = network.forward(img)
        e    = error.forward(label, y)
        # print(y)
        # print(e)
        # break  # remove this break------------------------------------------
        dedy = error.backward(e)
        # print(dedy.shape,"--------------")
        dedx = network.backward(dedy)

        # weight update
        network.update(lr)

        # update statistics
        training_loss = training_loss + e

    # break # remove this break -----------------------------------------------
    # cycle through the testing data
    for t in range(data_loader_test.num()):

        # data
        img, label = data_loader_test.get(t)

        # network forward pass and prediction
        y          = network.forward(img)
        prediction = (np.argmax(y)).astype(np.int32)

        # update statistics
        if (label == prediction):
            testing_correct = testing_correct + 1
            
    # epoch statistics
    elapsed_time_epoch = time.time() - start_time_epoch
    start_time_epoch   = time.time()
    epochs[epoch]      = epoch
    avg_loss[epoch]   = training_loss/data_loader_train.num()
    accuracy[epoch]    = 100.0*testing_correct/data_loader_test.num()
    print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f} accuracy = {4:5.2f}'.format(epoch, elapsed_time_epoch, lr, avg_loss[epoch], accuracy[epoch]), flush=True)

################################################################################
#
# DISPLAY
#
################################################################################

# plot of loss and accuracy vs epoch
fig1, ax1 = plt.subplots()
ax1.plot(epochs, avg_loss, color='red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Avg loss', color='red')
ax1.set_title('Avg Loss And Accuracy Vs Epoch')
ax2 = ax1.twinx()
ax2.plot(epochs, accuracy, color='blue')
ax2.set_ylabel('Accuracy %', color='blue')

# initialize the display predictions
predictions = np.zeros(DISPLAY_NUM, dtype=np.int32)

# cycle through the display data
for t in range(DISPLAY_NUM):

    # data
    img, label = data_loader_test.get(t)

    # network forward pass and prediction
    y              = network.forward(img)
    predictions[t] = (np.argmax(y)).astype(np.int32)

# plot of display examples
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for t in range(DISPLAY_NUM):
    img, label = data_loader_test.get(t)
    img        = img.reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, t + 1))
    ax[-1].set_title('True: ' + str(label) + ' xNN: ' + str(predictions[t]))
    plt.imshow(img, cmap='Greys')

# show figures
plt.show()

# # checking by value & reference.

# def set_list(list): 
#     list = ["A", "B", "C"] 
#     return list
  
# def add(list): 
#     list.append("D") 
#     return list
  
# my_list = ["E"] 
  
# print(set_list(my_list)) 
# print(my_list)
# print(add(my_list)) 
# print(my_list)

# t=np.ones(9)
# t.resize(3,3)

# print(t)