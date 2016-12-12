#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 23:39:10 2016

@author: rongdilin
"""
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import tensorflow as tf
import os
import time
import numpy as np
import string
from tensorflow.contrib import learn

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
#data clear: make all vobal lowercase, remove punctuation, number
def normalize_text(texts):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
#    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    
    return(texts)
    
#extract the author's id and text in xml file
#open xml file
tree = ET.ElementTree(file='/Users/rongdilin/Desktop/cse610/proj2/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
#get the author and text in the xml
author = []
text = []
for elem in tree.iter(tag = 'author'):
    author.append(elem.text)
for elem in tree.iter(tag = 'text'):
    text.append(elem.text)

#data clear    
#author = normalize_text(author)
#text = normalize_text(text)

author_text = [author, text]

#separation of predators in positive file
positive_dict = {}
negative_dict = {}
positive_file = []
negative_file = []
predators = []
f = open('/Users/rongdilin/Desktop/cse610/proj2/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r')
#remove '/n'
for line in f.readlines():
    predators.append(line.strip('\n'))

#positive dict: predator_id, text_list
#negative_dict: normal_id, text_list
for i in xrange(len(author)):
   count = 0
   for predator in predators: 
       if predator == author[i]:
           if(author[i] not in positive_dict):
               positive_dict[author[i]] = []
           positive_dict.get(author[i]).append(text[i])
           positive_file.append(text[i])
           count += 1
   if count == 0 : 
       if(author[i] not in negative_dict):
           negative_dict[author[i]] = []
       negative_dict.get(author[i]).append(text[i])
       negative_file.append(text[i])
# Generate labels
positive_labels = [[0, 1] for _ in positive_file]
negative_labels = [[1, 0] for _ in negative_file]
y = np.concatenate([positive_labels, negative_labels], 0)

#ats = zip(author, text)
#atDict = dict((author, text) for author, text in ats)

# Build vocabulary
length = []
for x in xrange (len(text)):
    if text[x] == None:
        length.append(0)
    else:
        length.append(len(text[x]))
max_document_length = max(length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(text)))

# Placeholders for input, output and dropout
#sequence_length – The length of our sentences. Remember that we padded all our sentences to have the same length (59 for our data set).
#num_classes – Number of classes in the output layer, two in our case (positive and negative).
#vocab_size – The size of our vocabulary. This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size].
#embedding_size – The dimensionality of our embeddings.
#filter_sizes – The number of words we want our convolutional filters to cover. We will have num_filters for each size specified here. For example, [3, 4, 5] 
#means that we will have filters that slide over 3, 4 and 5 words respectively, for a total of 3 * num_filters filters.
#num_filters – The number of filters per filter size (see above).
sequence_length = 59
num_classes = 2
vocab_size = len(vocab_processor.vocabulary_)
embedding_size = 128
filter_sizes = [3, 4, 5]
num_filters = 128

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
# Embedding layer
with tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

#convolution and max pooling layer    
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)
 
# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(3, pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
# SCORES AND PREDICTIONS
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
    loss = tf.reduce_mean(losses)
# Calculate Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

