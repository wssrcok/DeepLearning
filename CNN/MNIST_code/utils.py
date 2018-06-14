import h5py
import numpy as np
import tensorflow as tf
from layers.cnn_layers import *
from layers.fc_layers import *

def load_dataset():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    classes = 10
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_labels = one_hot_label(classes, train_labels)
    eval_data = mnist.test.images # Returns np.array
    eval_labels_old = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_labels = one_hot_label(classes, eval_labels_old)
    features = train_data.shape[0]

    train_data = train_data.reshape(55000,1,28,28)
    eval_data = eval_data.reshape(10000,1,28,28)
    return train_data, train_labels, eval_data, eval_labels, classes

def one_hot_label(classes, label):
    """
    reshape label to Sam prefered shape for mnist

    Arguments:
    label -- input label with shape (m,)

    Returns:
    new_label -- output label with shape (classes, 1, m)
    """
    m = label.shape[0]
    new_label = np.zeros((classes, m))
    for i in range(m):
        clas = label[i]
        new_label[clas,i] = 1
    return new_label

def eval_cnn_model(X, y, parameters, parameters_conv):
    
    m = X.shape[0]
    p = np.zeros((y.shape[0],m))
    
    # Forward propagation
    A, _ = two_conv_pool_layer_forward(X, parameters_conv)
    # flatten
    A = A.reshape(-1,7*7*64).T
    probas, _ = L_model_forward(A, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(probas.shape[0]):
        for j in range(probas.shape[1]):
            if probas[i,j] == np.max(probas[:,j]):
                p[i,j] = 1
            else:
                p[i,j] = 0
    match = 0
    for i in range(m):
        match += int(np.array_equal(p[:,i],y[:,i]))
    print("Accuracy: "  + str(np.sum(match*1.0/m)))
        
    return p

def predict_cnn(X, parameters, parameters_conv):
	m = X.shape[0]
	p = np.zeros((1,m))

	# Forward propagation
	A, _ = two_conv_pool_layer_forward(X, parameters_conv)
	# flatten
	A = A.reshape(-1,7*7*64).T
	probas, caches = L_model_forward(A, parameters)

	# convert probas to 0/1 predictions
	for i in range(probas.shape[0]):
		for j in range(probas.shape[1]):
			if probas[i,j] == np.max(probas[:,j]):
				p[0,j] = i
	print ("predictions: " + str(int(np.squeeze(p))))
	return

def eval_model_fc(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((y.shape[0],m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(probas.shape[0]):
        for j in range(probas.shape[1]):
            if probas[i,j] == np.max(probas[:,j]):
                p[i,j] = 1
            else:
                p[i,j] = 0
    
    #print results
    #print ("predictions: " + str(probas))
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    match = 0
    for i in range(m):
        match += int(np.array_equal(p[:,i],y[:,i]))
    print("Accuracy: "  + str(np.sum(match*1.0/m)))
        
    return p

def predict_fc(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(probas.shape[0]):
        for j in range(probas.shape[1]):
            if probas[i,j] == np.max(probas[:,j]):
                p[0,j] = i
    
    #print results
    #print ("predictions: " + str(probas))
    print ("predictions: " + str(int(np.squeeze(p))))
    #print ("true labels: " + str(y))
    return
