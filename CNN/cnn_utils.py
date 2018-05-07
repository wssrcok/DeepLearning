import numpy as np
from fc_utils import *
import matplotlib.pyplot as plt
from im2col import *
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(2)

def conv_forward(A_prev, W, b, hparameters):
	# W's shape used to be (f, f, n_C_prev, n_C)
	n_C, n_C_prev, f, f = W.shape
	# A_prev's shape used to be: (m, n_H_prev, n_W_prev, n_C_prev)
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

	A, old_Z = relu(Z)


	cache = (A_prev, W, b, hparameters, A_prev_col, old_Z)
	return A, cache


def pool_forward(A_prev, hparameters):
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
	A = A.transpose(2, 3, 1, 0)
	# now A is shape (m,n_C_prev, n_H, n_W)
	
	cache = (A_prev, A_prev_col, max_idx, hparameters)
	return A, cache

def conv_backward(dZ, cache):
	'''
	Arguments:
	dZ -- numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward(), output of conv_forward()

	Returns:
	dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
	           numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	dW -- gradient of the cost with respect to the weights of the conv layer (W)
	      numpy array of shape (f, f, n_C_prev, n_C)
	db -- gradient of the cost with respect to the biases of the conv layer (b)
	      numpy array of shape (1, 1, 1, n_C)
	'''
	#cache have the dimension: (A_prev, W, b, hparameters, A_prev_col)
	A_prev, W, b, hparameters, A_prev_col = cache
	# W.shape used to be (f,f,n_C_prev, n_C)
	n_C, n_C_prev, f, f = W.shape

	#dZ -- numpy array of shape (m, n_H, n_W, n_C) 
	# however, we want dZ to be (m, n_C, n_H, n_W) 

	db = np.sum(dZ, axis=(0, 2, 3))
	db = db.reshape(n_C, -1)
	dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
	dW = dZ_reshaped @ A_prev_col.T
	dW = dW.reshape(W.shape)

	stride = hparameters['stride']
	pad = hparameters['pad']

	W_reshape = W.reshape(n_C, -1)
	dA_prev_col = W_reshape.T @ dZ_reshaped
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
	(m, n_H, n_W, n_C) = dA.shape
	dA = dA.transpose(0,3,1,2)
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



def initialize_parameters_filter(filter_dim):
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
    W1 = np.random.randn(n_C1, n_C_prev1, f1, f1) * 0.01
    b1 = np.zeros((n_C1,1))
    n_C3, n_C_prev3, f3, f3 = filter_dim[1]
    W3 = np.random.randn(n_C3, n_C_prev3, f3, f3) * 0.01
    b3 = np.zeros((n_C3,1))
    parameters = {'W1':W1, 'b1':b1, 'W3':W3, 'b3':b3}
    return parameters

def two_conv_pool_layer_forward(input_layer, parameters):
    caches = []
    W1 = parameters['W1']
    b1 = parameters['b1']
    W3 = parameters['W3']
    b3 = parameters['b3']
    hparameters = {'stride': 1, 'pad': 2}
    A1, conv_cache1 = conv_forward(input_layer, W1, b1, hparameters)
    caches.append(conv_cache1)
    hparameters = {'f': 2, 'stride': 2}
    A2, pool_cache2 = pool_forward(A1, hparameters)
    caches.append(pool_cache2)
#     plt.imshow(A2[0,:,:,0]*5000)
    hparameters = {'stride': 1, 'pad': 2}
    A3, conv_cache3 = conv_forward(A2, W3, b3, hparameters)
    caches.append(conv_cache3)
    hparameters = {'f': 2, 'stride': 2}
    A4, pool_cache4 = pool_forward(A3, hparameters)
    #plt.imshow(A4[0,:,:,9])
    caches.append(pool_cache4)
    return A4, caches

def two_conv_pool_layer_backward(dA4, caches):
    '''
    Arguments:
    caches -- contains [conv_cache, pool_cache, conv_cache...]
    dA4 -- dJ/dA4 which is equal to dJ/dZ * dZ/dA4 from first FC layer
    '''
    conv_cache1, pool_cache2, conv_cache3, pool_cache4 = caches
    m = dA4.shape[0]
    assert(dA4.shape[1] == 3136)
    dA4 = dA4.reshape(m,7,7,64)
    conv_grads = {}
    dA3 = pool_backward(dA4, pool_cache4)
    dZ3 = relu_backward(dA3, conv_cache3[-1])
    dA2, dW3, db3 = conv_backward(dZ3, conv_cache3[0:-1])
    
    dA1 = pool_backward(dA2, pool_cache2)
    dZ1 = relu_backward(dA1,  conv_cache1[-1])
    dA0, dW1, db1 = conv_backward(dZ1, conv_cache1[0:-1])
    conv_grads["dW3"] = dW3
    conv_grads["db3"] = db3
    conv_grads["dW1"] = dW1
    conv_grads["db1"] = db1
    
    return conv_grads

def update_conv_parameters(parameters, grads, learning_rate):
    '''
    '''
    parameters['W3'] -= learning_rate * grads['dW3']
    parameters['b3'] -= learning_rate * grads['db3']
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    return parameters
    
def cnn_model(input_layer, Y, filter_dims, layers_dims, parameters = {}, parameters_conv = {}, batch_size = 64, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    if len(parameters) == 0:
        parameters_conv = initialize_parameters_filter(filter_dims)
        parameters = initialize_parameters_deep(layers_dims)
    m = input_layer.shape[0]
    num_batchs = m // batch_size
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        for j in range(num_batchs):
            # conv forward
            input_batch = input_layer[j*batch_size:(j+1)*batch_size]
            A4, conv_caches = two_conv_pool_layer_forward(input_batch, parameters_conv)
            #flatten
            A4 = A4.reshape(batch_size,-1).T
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            # AL is the output and caches contains Z, A, W, b for each layer
            AL, caches = L_model_forward(A4, parameters)
            # Compute cost.
            cost = compute_cost(AL,Y[:, j*batch_size:(j+1)*batch_size])
            # Backward propagation.
            grads = L_model_backward(AL, Y[:, j*batch_size:(j+1)*batch_size], caches)
            # unflatten
            dA4 = grads['dA0'].T
            conv_grads = two_conv_pool_layer_backward(dA4, conv_caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
            parameters_conv = update_conv_parameters(parameters_conv, conv_grads, learning_rate)
            # Print the cost every 30 training example
            if print_cost:# and j % 300 == 0:
                print ("Cost after iteration %i, batch %i: %f" %(i, j, cost))
            if print_cost and j % 1 == 0:
                costs.append(cost)
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        # if print_cost and i % 5 == 0:
        #     costs.append(cost)
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters, parameters_conv, grads, conv_grads

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
def round8(parameters_conv, parameters):
	for k,v in parameters_conv.items():
		np.around(v, decimals=2)
	for k,v in parameters.items():
		np.around(v, decimals=2)

