# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:41:39 2018

@author: jaydeep thik
"""

import tensorflow as tf
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from utility import load_dataset, encode_one_hot

X_train, X_test, y_train, y_test, classes = load_dataset()

#plt.imshow(X_train[0])
#print(y_train[:,0])

X_train, X_test = X_train/255., X_test/255.
y_train = encode_one_hot(y_train, len(classes))
y_test = encode_one_hot(y_test, len(classes))

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
       
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size]
       
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[(k+1)*mini_batch_size:]
        mini_batch_Y =shuffled_Y[(k+1)*mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def create_plceholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name="X")
    y = tf.placeholder(tf.float32, [None, n_y], name="y")
    
    return X, y

def initialize_params():
    
    W1 = tf.get_variable("W1", [4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [4,4,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [2,2,16,32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {"W1":W1,"W2":W2,"W3":W3}
    
    return parameters

def forward_prop(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    Z1 = tf.nn.conv2d(X, W1, [1,1,1,1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,4,4,1],strides=[1,4,4,1], padding="SAME" )
    
    Z2 = tf.nn.conv2d(P1, W2,strides=[1,1,1,1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")
    
    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding="SAME")
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    P3 = tf.contrib.layers.flatten(P3)
    
    Z4 = tf.contrib.layers.fully_connected(P3,6, activation_fn=None)
    
    
    return Z4

def compute_cost(Z4, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4,labels= y))

def model(X_train, y_tarin, X_test, y_test, lr=0.009, num_epoches=200, mini_batch_size=32,print_cost=True):
    
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = y_train.shape[1]
    total_cost=[]
    
    X, y = create_plceholder(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_params()
    Z4 = forward_prop(X, parameters)
    cost = compute_cost(Z4, y)
    
    #backprop
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epoches):
            mini_batch_cost = 0.
            num_mini_batches = int(m/mini_batch_size)
            minibatches = random_mini_batches(X_train, y_train, mini_batch_size)
            
            for batch in minibatches:
                (X_batch, y_batch) = batch
                _, temp_cost = sess.run([optimizer, cost],feed_dict={X:X_batch, y:y_batch})
                
                mini_batch_cost+=temp_cost/num_mini_batches
                
            if (print_cost):
                total_cost.append(mini_batch_cost)
                if(epoch%5==0):
                    print("epoch :",epoch," cost :", mini_batch_cost)
                    
        plt.plot(np.squeeze(total_cost))
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.title("learning rate = "+str(lr))
        plt.show()

        predict_op = tf.arg_max(Z4, 1, name="predict_op")
        correct_prediction = tf.equal(predict_op, tf.arg_max(y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        train_accuracy = accuracy.eval({X:X_train, y:y_train})
        test_accuracy = accuracy.eval({X:X_test,   y:y_test})
        print("Training:",train_accuracy," Test :", test_accuracy)
        saver.save(sess, "./my-test-model")
        return train_accuracy, test_accuracy, parameters
    
    