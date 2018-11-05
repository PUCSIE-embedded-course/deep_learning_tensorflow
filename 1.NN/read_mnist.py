from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

#read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

##################################
# print the shape of the dataset #
##################################

#55000 training images with the size of 784 (28x28)
print(train_img.shape)

#answers (label) of these 55000 images.
#the label is expressed as following format: e.g., 3 is stored as [0, 0, 0, 1, 0, 0, 0, 0, 0 , 0],
#the size of it is therefore 10.
print(train_label.shape)

#10000 testing images
print(test_img.shape)
#answer of these 10000 testing image
print(test_label.shape)

##########################################################
# plot a image from the MNIST dataset and read its value #
##########################################################

#print(train_img[1, :])
print("The image is " + str(np.argmax(train_label[1, :])))

#show a image with matlabplot
img = np.reshape(train_img[1, :], (28, 28))     #convert data format
plt.matshow(img, cmap = plt.get_cmap('gray')) #set stroke color
plt.show()                                    #show the image
