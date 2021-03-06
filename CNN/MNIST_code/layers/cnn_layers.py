import numpy as np
from layers.quantize import *
from layers.im2col import *
from layers.general_layers import relu, relu_backward

np.random.seed(2)

def conv_forward(A_prev, W, b, hparameters = {'stride': 1, 'pad': 2}, truncate = 0):

	n_C, n_C_prev, f, f = W.shape
	m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape

	stride = hparameters['stride']
	pad = hparameters['pad']
	A_prev_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
	W_col = W.reshape(n_C, -1)

	Z = W_col @ A_prev_col + b
	n_H_shape = (n_H_prev - f + 2 * pad) / stride + 1
	n_W_shape = (n_W_prev - f + 2 * pad) / stride + 1
	n_H_shape, n_W_shape = int(n_H_shape), int(n_W_shape)

	Z = Z.reshape(n_C, n_H_shape, n_W_shape, m)
	Z = Z.transpose(3, 0, 1, 2)
	# now Z.shape is (m, n_C, n_H_shape, n_W_shape)

	cache = (A_prev, W, b, hparameters, A_prev_col)
	return Z, cache


def pool_forward(A_prev, hparameters = {'f': 2, 'stride': 2}):
	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	(m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
	A_prev_reshaped = A_prev.reshape(m * n_C_prev, 1, n_H_prev, n_W_prev)

	f = hparameters["f"]
	stride = hparameters["stride"]
	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	A_prev_col = im2col_indices(A_prev_reshaped, f, f, padding=0, stride=stride)

	# Next, at each possible patch location, i.e. at each column, we're taking the max index
	max_idx = np.argmax(A_prev_col, axis=0)

	# Finally, we get all the max value at each column
	# The result will be 1x9800
	A = A_prev_col[max_idx, range(max_idx.size)]

	# Reshape to the output size: 14x14x5x10
	A = A.reshape(n_H_prev//f, n_W_prev//f, m, n_C_prev)

	# Transpose to get 5x10x14x14 output
	A = A.transpose(2, 3, 0, 1)
	# now A is shape (m,n_C_prev, n_H, n_W)
	
	cache = (A_prev, A_prev_col, max_idx, hparameters)
	return A, cache

def conv_backward(dZ, cache):
	'''
	Arguments:
	dZ -- numpy array of shape (m, n_C, n_H, n_W) 
	cache -- cache of values needed for the conv_backward(), output of conv_forward()

	Returns:
	dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
			   numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
	dW -- gradient of the cost with respect to the weights of the conv layer (W)
		  numpy array of shape (n_C_prev, n_C, f, f)
	db -- gradient of the cost with respect to the biases of the conv layer (b)
		  numpy array of shape (n_C, 1)
	'''

	A_prev, W, b, hparameters, A_prev_col = cache
	n_C, n_C_prev, f, f = W.shape 

	db = np.sum(dZ, axis=(0, 2, 3))
	db = db.reshape(n_C, -1)
	dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
	dW = dZ_reshaped @ A_prev_col.T
	dW = dW.reshape(W.shape)

	stride = hparameters['stride']
	pad = hparameters['pad']

	W_reshape = W.reshape(n_C, -1)
	dA_prev_col = W_reshape.T @ dZ_reshaped
	# turn column to image
	dA_prev = col2im_indices(dA_prev_col, A_prev.shape, f, f, padding=pad, stride=stride)
	return dA_prev, dW, db



def pool_backward(dA, cache):
	# X_col and max_idx are the intermediate variables from the forward propagation step

	# Suppose our output from forward propagation step is 5x10x14x14
	# We want to upscale that back to 5x10x28x28, as in the forward step

	# 4x9800, as in the forward step
	(A_prev, A_prev_col, max_idx, hparameters) = cache
	dA_prev_col = np.zeros_like(A_prev_col)

	f = hparameters["f"]
	stride = hparameters["stride"]

	#dA.shape is m,n_H,n_W,n_C
	(m, n_C, n_H, n_W) = dA.shape
	# 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
	# Transpose step is necessary to get the correct arrangement
	dA_flat = dA.transpose(2, 3, 0, 1).ravel()

	# Fill the maximum index of each column with the gradient

	# Essentially putting each of the 9800 grads
	# to one of the 4 row in 9800 locations, one at each column
	dA_prev_col[max_idx, range(max_idx.size)] = dA_flat

	# We now have the stretched matrix of 4x9800, then undo it with col2im operation
	# dX would be 50x1x28x28
	dA_prev = col2im_indices(dA_prev_col, (m * n_C, 1, n_H*f, n_W*f), f, f, padding=0, stride=stride)

	dA_prev = dA_prev.reshape(A_prev.shape)

	return dA_prev

def batchnorm_forward_conv(X, gamma, beta):
	N,C,H,W = X.shape
	mu = np.mean(X, axis=(0,2,3)).reshape(1,C,1,1)
	var = np.var(X, axis=(0,2,3)).reshape(1,C,1,1)
	gamma = gamma.reshape(1,C,1,1)
	beta = beta.reshape(1,C,1,1)

	X_norm = (X - mu) / np.sqrt(var + 1e-8)
	out = gamma * X_norm + beta

	cache = (X, X_norm, mu, var, gamma, beta)

	return out, cache, mu, var

def batchnorm_backward_conv(dout, cache):
	X, X_norm, mu, var, gamma, beta = cache

	N,C,H,W = X.shape

	X_mu = X - mu
	std_inv = 1. / np.sqrt(var + 1e-8)
	dX_norm = dout * gamma
	dvar = np.sum(dX_norm * X_mu, axis=(0,2,3)).reshape(1,C,1,1) * -.5 * std_inv**3
	dmu = np.sum(dX_norm * -std_inv, axis=(0,2,3)).reshape(1,C,1,1) + dvar * np.mean(-2. * X_mu, axis=(0,2,3)).reshape(1,C,1,1)
	dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / C) + (dmu / C)
	dgamma = np.sum(dout * X_norm, axis=(0,2,3)).reshape(C,1)
	dbeta = np.sum(dout, axis=(0,2,3)).reshape(C,1)

	return dX, dgamma, dbeta



def conv_pool_forward_general(input_feature, parameters, truncate = 0):
	caches = []
	L = len(parameters) // 2
	A = input_feature
	for l in range(1, L+1):
		A_prev = A		
		conv_out, conv_cache = conv_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 
									        truncate = truncate)

		relu_out, relu_cache = relu(conv_out, truncate = truncate)
		A, pool_cache = pool_forward(relu_out)

		caches.append((conv_cache, relu_cache, pool_cache))

	return A, caches

def conv_pool_backward_general(dAL, caches):
	'''
	'''
	L = len(caches) # number of layers
	grads = {}
	grads["dA" + str(L)] = dAL

	for l in reversed(range(L)): # L-2 ... 0
		conv_cache, relu_cache, pool_cache = caches[l]
		dP = pool_backward(grads["dA" + str(l+1)], pool_cache)
		dR = relu_backward(dP, relu_cache)
		dA, dW, db = conv_backward(dR, conv_cache)
		grads["dA" + str(l)] = dA
		grads["dW" + str(l+1)] = dW
		grads["db" + str(l+1)] = db
	
	return grads

def two_conv_pool_layer_forward(input_feature, parameters, truncate = 0):
	caches = []
	W1 = parameters['W1']
	b1 = parameters['b1']
	W3 = parameters['W3']
	b3 = parameters['b3']
	# g1 = parameters['g1']
	# be1 = parameters['be1']
	# g3 = parameters['g3']
	# be3 = parameters['be3']
	#----------------------------Conv_BN_relu_Pool_1--------------------------------------
	Z1, conv_cache1 = conv_forward(input_feature, W1, b1, truncate = truncate)
	caches.append(conv_cache1)

	# Zn1, bn_cache1, _, _ = batchnorm_forward_conv(Z1, g1, be1)
	# caches.append(bn_cache1)

	A1, relu_cache1 = relu(Z1, truncate = truncate)
	caches.append(relu_cache1)

	A2, pool_cache2 = pool_forward(A1)
	caches.append(pool_cache2)
	#---------------------------Conv_BN_relu_Pool_2--------------------------------------	
	Z3, conv_cache3 = conv_forward(A2, W3, b3, truncate = truncate)
	caches.append(conv_cache3)

	# Zn3, bn_cache3, _, _ = batchnorm_forward_conv(Z3, g3, be3)
	# caches.append(bn_cache3)

	A3, relu_cache3 = relu(Z3, truncate = truncate)
	caches.append(relu_cache3)

	A4, pool_cache4 = pool_forward(A3)
	caches.append(pool_cache4)
	#------------------------------------------------------------------------------------
	return A4, caches

def two_conv_pool_layer_backward(dA4, caches):
	'''
	Arguments:
	caches -- contains [conv_cache, pool_cache, conv_cache...]
	dA4 -- dJ/dA4 which is equal to dJ/dZ * dZ/dA4 from first FC layer
	'''
	conv_cache1,  relu_cache1, pool_cache2, \
	conv_cache3,  relu_cache3, pool_cache4 = caches	
	conv_grads = {}
	#------------------------Pool_relu_BN_Conv_2----------------------------------------
	dA3 = pool_backward(dA4, pool_cache4)
	dZ3 = relu_backward(dA3, relu_cache3)
	# dZn3, dgamma3, dbeta3 = batchnorm_backward_conv(dZ3, bn_cache3)
	dA2, dW3, db3 = conv_backward(dZ3, conv_cache3)
	#------------------------Pool_relu_BN_Conv_1----------------------------------------
	dA1 = pool_backward(dA2, pool_cache2)
	dZ1 = relu_backward(dA1,  relu_cache1)
	# dZn1, dgamma1, dbeta1 = batchnorm_backward_conv(dZ1, bn_cache1)
	dA0, dW1, db1 = conv_backward(dZ1, conv_cache1)
	#------------------------------------------------------------------------------------
	conv_grads["dW3"] = dW3
	conv_grads["db3"] = db3
	conv_grads["dW1"] = dW1
	conv_grads["db1"] = db1
	# conv_grads["dg3"] = dgamma3
	# conv_grads["dbe3"] = dbeta3
	# conv_grads["dg1"] = dgamma1
	# conv_grads["dbe1"] = dbeta1
	
	
	return conv_grads
