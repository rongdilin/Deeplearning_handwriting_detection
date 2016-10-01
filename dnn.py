# start tensorflow interactiveSession
import tensorflow as tf
from PIL import Image
import numpy as np
import os, sys

'''the size of image is 100 * 200'''

train_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/images"
train_listFileName = os.popen('find ' + train_path)
train_fileAddress = train_listFileName.readlines()

#create a matrix containing files numbers of matrix that size is [100, 200] all zero
train_data = [[0 for i in range(20000)] for j in range(train_fileAddress)]
train_label = range(1, len(train_fileAddress))

#print(data)
#len(fileAddress)
for i in range(0,train_fileAddress):
    #the end of each path has a /n so remove it 
    train_imagepath = train_fileAddress[i+2][:-1]
    train_img = np.array(Image.open(train_imagepath), np.float32)
    
    #initial a new matrix
    train_matrix = np.zeros((100, 200))
    train_matrix = train_matrix[train_img.shape[0], train_img.shape[1]] + train_img
    train_matrix[train_img.shape[0] : 100, train_img.shape[1] : 200] += 255
    #flat matrix
    train_matrix = np.ravel(train_matrix)
    
    #regulariztion
    train_data[i] = np.multiply(train_matrix, 1.0 / 255.0)
    
#print(data)

test_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/test"
test_listFileName = os.popen('find ' + test_path)
test_fileAddress = test_listFileName.readlines()
test_data = [[0 for i in range(20000)] for j in range(test_fileAddress)]
test_label = range(1, len(test_fileAddress))

for i in range(0,test_fileAddress):
    test_imagepath = test_fileAddress[i+2][:-1]
    test_img = np.array(Image.open(test_imagepath), np.float32)
    
    #initial a new matrix
    test_matrix = np.zeros((100, 200))
    test_matrix = test_matrix[test_img.shape[0], test_img.shape[1]] + test_img
    test_matrix[test_img.shape[0] : 100, test_img.shape[1] : 200] += 255
    #flat matrix
    test_matrix = np.ravel(test_matrix)
    
    #regulariztion
    test_data[i] = np.multiply(test_matrix, 1.0 / 255.0)
    


#######################tensorflow###########

sess = tf.InteractiveSession()

# weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def minibatch(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
# Create the model
# placeholder
#20000 is the dimensionality of a single flattened 100 by 200 pixel MNIST image, 
#and None indicates that the first dimension, corresponding to the batch size, can be of any size. 
x = tf.placeholder("float", [None, 20000])
#The target output classes y_ will also consist of a 2d tensor, where each row is a one-hot 
#2-dimensional vector indicating which digit class (1 or 0) the corresponding whether character
#belongs to the same writer.
y_ = tf.placeholder("float", [None, 2])

# variables
W = tf.Variable(tf.zeros([20000,2]))
b = tf.Variable(tf.zeros([2]))
#initial
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 100, 200, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
	batch = minibatch(train_data, train_label, 50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print "step %d, train accuracy %g" %(i, train_accuracy)
	train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={x:test_data, y_:test_label, keep_prob:1.0})