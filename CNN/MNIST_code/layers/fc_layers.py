import numpy as np
from layers.quantize import *
from layers.general_layers import *

def batchnorm_forward_fc(X, gamma, beta):
	mu = np.mean(X, axis=(1)).reshape(-1,1)
	var = np.var(X, axis=(1)).reshape(-1,1)
	X_norm = (X - mu) / np.sqrt(var + 1e-8)
	out = gamma * X_norm + beta

	cache = (X, X_norm, mu, var, gamma, beta)

	return out, cache, mu, var

def batchnorm_backward_fc(dout, cache):
	X, X_norm, mu, var, gamma, beta = cache

	N,D = X.shape
	X_mu = X - mu
	std_inv = 1. / np.sqrt(var + 1e-8)

	dX_norm = dout * gamma
	dvar = np.sum(dX_norm * X_mu, axis=(1)).reshape(-1,1) * -.5 * std_inv**3
	dmu = np.sum(dX_norm * -std_inv, axis=(1)).reshape(-1,1) + dvar * np.mean(-2. * X_mu, axis=(0))

	dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
	dgamma = np.sum(dout * X_norm, axis=(1)).reshape(-1,1)
	dbeta = np.sum(dout, axis=(1)).reshape(-1,1)

	return dX, dgamma, dbeta

def linear_forward(A, W, b, truncate = 0):
	"""
	Implement the linear part of a layer's forward propagation.

	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	
	### START CODE HERE ### (≈ 1 line of code)
	Z = np.dot(W,A)+b
	### END CODE HERE ###
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation, truncate = 0):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python dictionary containing "linear_cache" and "activation_cache";
			 stored for computing the backward pass efficiently
	"""
	
	if activation == "sigmoid":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		### START CODE HERE ### (≈ 2 lines of code)
		Z, linear_cache = linear_forward(A_prev,W,b, truncate = truncate)
		#Zn, bn_cache, _, _ = batchnorm_forward_fc(Z, g, be)
		A, activation_cache = sigmoid(Zn)
		### END CODE HERE ###
	
	elif activation == "relu":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		### START CODE HERE ### (≈ 2 lines of code)
		Z, linear_cache = linear_forward(A_prev,W,b,truncate = truncate)
		#Zn, bn_cache, _, _ = batchnorm_forward_fc(Z, g, be)
		A, activation_cache = relu(Z, truncate = truncate)
		### END CODE HERE ###
	elif activation == "softmax":
		Z, linear_cache = linear_forward(A_prev,W,b,truncate = truncate)
		bn_cache = 0
		A, activation_cache = softmax(Z)
		#print(A[:,3])
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	#cache = (linear_cache, bn_cache, activation_cache)
	cache = (linear_cache, activation_cache)
	return A, cache

def L_model_forward(X, parameters, truncate = 0):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	
	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
				every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""

	caches = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network
	
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A 
		### START CODE HERE ### (≈ 2 lines of code)
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],
											 parameters['b' + str(l)], "relu")
		# A is truncated in the function
		caches.append(cache)
		### END CODE HERE ###
	
	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	### START CODE HERE ### (≈ 2 lines of code)
	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
										  parameters['b' + str(L)], "softmax", truncate = truncate)
	# A is truncated in the function
	caches.append(cache)
	### END CODE HERE ###
			
	return AL, caches

def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]

	### START CODE HERE ### (≈ 3 lines of code)
	dW = 1/m*np.dot(dZ,A_prev.T)
	db = 1/m*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T,dZ)
	### END CODE HERE ###
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, Y, Y_hat, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		### START CODE HERE ### (≈ 2 lines of code)
		dZ = relu_backward(dA, activation_cache)
		#dZn, dg, dbe = batchnorm_backward_fc(dZ, bn_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)
		### END CODE HERE ###
		
	elif activation == "sigmoid":
		### START CODE HERE ### (≈ 2 lines of code)
		dZ = sigmoid_backward(dA, activation_cache)
		#dZn, dg, dbe = batchnorm_backward_fc(dZ, bn_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)
		### END CODE HERE ###
	elif activation == "softmax":
		### START CODE HERE ### (≈ 2 lines of code)
		dZ = softmax_backward(Y, Y_hat)
		# dg = 0
		# dbe = 0
		dA_prev, dW, db = linear_backward(dZ,linear_cache)
		### END CODE HERE ###
	
	return dA_prev, dW, db#, dg, dbe

def L_model_backward(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# Initializing the backpropagation
	### START CODE HERE ### (1 line of code)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
	### END CODE HERE ###
	
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	### START CODE HERE ### (approx. 2 lines)
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, Y, AL, 'softmax')
	### END CODE HERE ###
	
	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		### START CODE HERE ### (approx. 5 lines)
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, Y, AL, 'relu')
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
		# grads["dg" + str(l + 1)] = dg_temp
		# grads["dbe" + str(l + 1)] = dbe_temp
		### END CODE HERE ###

	return grads

def compute_cost(AL, Y, cost_function = 'softmax_cross_entropy'):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (10, number of examples)
	Y -- true "label" vector (for example: [1,0,0,...,0] as 0
		 [0,1,0,...,0] as 1), shape (classes, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m = Y.shape[1]

	# Compute loss from aL and y.
	### START CODE HERE ### (≈ 1 lines of code)
	if cost_function == 'softmax_cross_entropy':
		cost = (-1/m) * np.sum(Y*np.log(AL))
	elif cost_function == 'sigmoid_cross_entropy':
		cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
	
	### END CODE HERE ###
	
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost