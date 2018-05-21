import numpy as np
from layers.quantize import *

def initialize_parameters_filter(filter_dim, truncate = 0):
	'''
	building a filter
	Arguments:
	filter_dim -- dimension of filter:[(f,f,n_C_prev, n_C),(f,f,n_C_prev, n_C)]
	Returns:
	W -- 
	caches -- list of caches containing:
				every cache of linear_act
	'''
	n_C1, n_C_prev1, f1, f1 = filter_dim[0]
	W1 = np.random.randn(n_C1, n_C_prev1, f1, f1) * 0.25
	b1 = np.zeros((n_C1,1))
	n_C3, n_C_prev3, f3, f3 = filter_dim[1]
	W3 = np.random.randn(n_C3, n_C_prev3, f3, f3) * 0.25
	b3 = np.zeros((n_C3,1))
	g1 = 1
	g3 = 1
	be1 = 0
	be3 = 0
	if truncate:
		W1 = truncate_weights(W1, truncate)
		b1 = truncate_weights(b1, truncate)
		W3 = truncate_weights(W3, truncate)
		b3 = truncate_weights(b3, truncate)
	#print(W1[0,0])
	parameters = {'W1':W1, 'b1':b1, 'W3':W3, 'b3':b3, 
				  'g1':g1, 'g3':g3, 'be1':be1, 'be3':be3}
	return parameters

def update_conv_parameters(parameters, grads, learning_rate, truncate = 0):
	'''
	'''
	# print('weights on third layer first training example first channel:\n', parameters['W3'][0,0])
	# print('learning rate:', learning_rate)
	# print('gradent for this weights:\n', grads['dW3'][0,0])
	parameters['W3'] -= learning_rate * grads['dW3']
	parameters['b3'] -= learning_rate * grads['db3']
	parameters['W1'] -= learning_rate * grads['dW1']
	parameters['b1'] -= learning_rate * grads['db1']
	# parameters['g1'] -= learning_rate * grads['dg1']
	# parameters['g3'] -= learning_rate * grads['dg3']
	# parameters['be1'] -= learning_rate * grads['dbe1']
	# parameters['be3'] -= learning_rate * grads['dbe3']
	# print('before subtract:\n', learning_rate * grads['dW3'][0,0])
	# print('weights after update:\n', parameters['W3'][0,0])
	if truncate:
		parameters['W3'] = truncate_weights(parameters['W3'], truncate)
		parameters['b3'] = truncate_weights(parameters['b3'], truncate)
		parameters['W1'] = truncate_weights(parameters['W1'], truncate)
		parameters['b1'] = truncate_weights(parameters['b1'], truncate)
		# parameters['g1'] = truncate_weights(parameters['g1'], truncate)
		# parameters['be1'] = truncate_weights(parameters['ge1'], truncate)
		# parameters['g3'] = truncate_weights(parameters['g3'], truncate)
		# parameters['be3'] = truncate_weights(parameters['ge3'], truncate)

		# print('weights after truncate:\n', parameters['W3'][0,0])
	return parameters

def initialize_parameters_deep(layer_dims, truncate = 0):
	"""
	Arguments:
	layer_dims -- python array (list) containing the dimensions of each layer in our network
	
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""
	
	np.random.seed(2)
	parameters = {}
	L = len(layer_dims)            # number of layers in the network

	for l in range(1, L):
		### START CODE HERE ### (≈ 2 lines of code)
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
		parameters['g' + str(l)] = 1
		parameters['be' + str(l)] = 0
		### END CODE HERE ###
		
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	if truncate:
		parameters["W" + str(l)] = truncate_weights(parameters["W" + str(l)], truncate)
		parameters["b" + str(l)] = truncate_weights(parameters["b" + str(l)], truncate)
		parameters["g" + str(l)] = truncate_weights(parameters["g" + str(l)], truncate)
		parameters["be" + str(l)] = truncate_weights(parameters["be" + str(l)], truncate)
		
	return parameters

def update_parameters(parameters, grads, learning_rate, truncate = 0):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	"""
	
	L = len(parameters) // 4 # number of layers in the neural network

	# Update rule for each parameter. Use a for loop.
	### START CODE HERE ### (≈ 3 lines of code)
	for l in range(L):
		parameters["W" + str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
		parameters["b" + str(l+1)] -= learning_rate * grads['db'+str(l+1)]
		parameters["g" + str(l+1)] -= learning_rate * grads['dg'+str(l+1)]
		parameters["be" + str(l+1)] -= learning_rate * grads['dbe'+str(l+1)]
		if truncate:
			parameters["W" + str(l+1)] = truncate_weights(parameters["W" + str(l+1)], truncate)
			parameters["b" + str(l+1)] = truncate_weights(parameters["b" + str(l+1)], truncate)
			parameters["g" + str(l+1)] = truncate_weights(parameters["g" + str(l+1)], truncate)
			parameters["be" + str(l+1)] = truncate_weights(parameters["be" + str(l+1)], truncate)
	### END CODE HERE ###
	return parameters