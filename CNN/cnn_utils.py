import numpy as np
from fc_utils import *
import matplotlib.pyplot as plt
from im2col import *
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    ### END CODE HERE ###
    
    return X_pad

# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    ### END CODE HERE ###

    return Z

def conv_forward(A_prev, W, b, hparameters):
	# W's shape used to be (f, f, n_C_prev, n_C)
	W = W.transpose(3,2,1,0)
	n_C, n_C_prev, f, f = W.shape
	# A_prev's shape used to be: (m, n_H_prev, n_W_prev, n_C_prev)
	A_prev = A_prev.transpose(0,3,1,2)
	m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape

	stride = hparameters['stride']
	pad = hparameters['pad']

	n_H_shape = (n_H_prev - f + 2 * pad) / stride + 1
	n_W_shape = (n_W_prev - f + 2 * pad) / stride + 1

	if not n_H_shape.is_integer() or not n_W_shape.is_integer():
		raise Exception('Invalid output dimension!')
	n_H_shape, n_W_shape = int(n_H_shape), int(n_W_shape)
	assert(A_prev.shape == (m,1,28,28) or A_prev.shape == (m,32,14,14))
	assert(pad == 2 and stride == 1)
	A_prev_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
	W_col = W.reshape(n_C, -1)

	b = b.reshape(n_C,1)
	Z = W_col @ A_prev_col + b
	Z = Z.reshape(n_C, n_H_shape, n_W_shape, m)
	Z = Z.transpose(3, 0, 1, 2)
	# now Z.shape is (m, n_C, n_H_shape, n_W_shape)

	A, old_Z = relu(Z)

	# turn every shape back to what I learned from coursera.
	W = W.transpose(3,2,1,0)
	A_prev = A_prev.transpose(0,2,3,1)
	b = b.reshape(1,1,1,n_C)
	A = A.transpose(0,2,3,1)
	old_Z = old_Z.transpose(0,2,3,1)

	cache = (A_prev, W, b, hparameters, old_Z)

	return A, cache

# def conv_forward(A_prev, W, b, hparameters):
#     """
#     Implements the forward propagation for a convolution function
    
#     Arguments:
#     A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
#     b -- Biases, numpy array of shape (1, 1, 1, n_C)
#     hparameters -- python dictionary containing "stride" and "pad"
        
#     Returns:
#     Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
#     cache -- cache of values needed for the conv_backward() function
#     """
    
#     ### START CODE HERE ###
#     # Retrieve dimensions from A_prev's shape (≈1 line)  
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
#     # Retrieve dimensions from W's shape (≈1 line)
#     (f, f, n_C_prev, n_C) = W.shape
    
#     # Retrieve information from "hparameters" (≈2 lines)
#     stride = hparameters['stride']
#     pad = hparameters['pad']
    
#     # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
#     n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
#     n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
#     # Initialize the output volume Z with zeros. (≈1 line)
#     Z = np.zeros((m, n_H, n_W, n_C));
    
#     # Create A_prev_pad by padding A_prev
#     A_prev_pad = zero_pad(A_prev, pad)
    
#     for i in range(0,m):                               # loop over the batch of training examples
#         a_prev_pad = A_prev_pad[i]                   # Select ith training example's padded activation
#         for h in range(0,n_H):                           # loop over vertical axis of the output volume
#             for w in range(0,n_W):                       # loop over horizontal axis of the output volume
#                 for c in range(0,n_C):                   # loop over channels (= #filters) of the output volume
                    
#                     # Find the corners of the current "slice" (≈4 lines)
#                     vert_start = h * stride
#                     vert_end = h * stride + f
#                     horiz_start = w * stride
#                     horiz_end = w * stride + f
                    
#                     # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
#                     a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
#                     # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
#                     Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                            
#     # activation
#     A, old_Z = relu(Z)       
#     ### END CODE HERE ###
    
#     # Making sure your output shape is correct
#     assert(A.shape == (m, n_H, n_W, n_C))
    
#     # Save information in "cache" for the backprop
#     cache = (A_prev, W, b, hparameters, old_Z)
    
#     return A, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    ### END CODE HERE ###
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    
    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)                           
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    assert(A_prev_pad.shape == (m, n_H_prev+2*pad, n_W_prev+2*pad, n_C_prev))
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        assert(a_prev_pad.shape == (n_H_prev+2*pad, n_W_prev+2*pad, n_C_prev))
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    assert(a_slice.shape == (f,f,n_C_prev))
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    
    ### START CODE HERE ### (≈1 line)
    mask = (x == np.max(x))
    ### END CODE HERE ###
    
    return mask

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz/(n_H*n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape)*average
    ### END CODE HERE ###
    
    return a

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i,h,w,c]
                        
                    elif mode == "average":
                        
                        # Get the value a from dA (≈1 line)
                        da = dA[i]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f,f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da[h,w,c], shape)
                        
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
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
    f1, f1, n_C_prev1, n_C1 = filter_dim[0]
    W1 = np.random.randn(f1,f1,n_C_prev1, n_C1) * 0.01
    b1 = np.zeros((1,1,1,n_C1))
    f3, f3, n_C_prev3, n_C3 = filter_dim[1]
    W3 = np.random.randn(f3,f3,n_C_prev3, n_C3) * 0.01
    b3 = np.zeros((1,1,1,n_C3))
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
    dZ3 = relu_backward(dA3, conv_cache3[4])
    dA2, dW3, db3 = conv_backward(dZ3, conv_cache3[0:4])
    
    dA1 = pool_backward(dA2, pool_cache2)
    dZ1 = relu_backward(dA1,  conv_cache1[4])
    dA0, dW1, db1 = conv_backward(dZ1, conv_cache1[0:4])
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
            A4 = A4.reshape(-1,7*7*64).T
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

