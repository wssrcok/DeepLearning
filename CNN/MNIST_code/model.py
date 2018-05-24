import numpy as np
from layers.cnn_layers import *
from layers.fc_layers import *
from layers.weights import *
import matplotlib.pyplot as plt

def cnn_model(input_layer, Y, filter_dims, layers_dims, truncate = 0, parameters = {}, parameters_conv = {}, batch_size = 64, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    
    Returns:
    """

    #np.random.seed(1)
    costs = []             # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    if len(parameters) == 0:
        parameters_conv = initialize_parameters_filter(filter_dims, truncate = truncate)
        parameters = initialize_parameters_deep(layers_dims, truncate = truncate)
    m = input_layer.shape[0]
    num_batchs = m // batch_size
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        for j in range(num_batchs):
            # conv forward
            input_batch = input_layer[j*batch_size:(j+1)*batch_size]
            A4, conv_caches = two_conv_pool_layer_forward(input_batch, parameters_conv, truncate = truncate)
            #flatten
            A4 = A4.reshape(batch_size,-1).T
            # AL is the output and caches contains Z, A, W, b for each layer
            AL, caches = L_model_forward(A4, parameters, truncate = truncate)
            # prevent divide by zero occur.
            if truncate:
	            AL = np.where(AL == 0,1/256,AL)
	            AL = np.where(AL == 1,255/256,AL)
            # Compute cost.
            cost = compute_cost(AL,Y[:, j*batch_size:(j+1)*batch_size])
            # Backward propagation.
            grads = L_model_backward(AL, Y[:, j*batch_size:(j+1)*batch_size], caches)
            # unflatten
            dA4 = grads['dA0'].T
            conv_grads = two_conv_pool_layer_backward(dA4, conv_caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate, truncate = truncate)
            parameters_conv = update_conv_parameters(parameters_conv, conv_grads, learning_rate, truncate = truncate)
            if print_cost: # and j % 1 == 0:
                print ("Cost after iteration %i, batch %i: %f" %(i, j, cost))
            if print_cost:# and j % 1 == 0:
                costs.append(cost)
        if print_cost:# and i % 1 == 0:
            print('Epoch %i, Done!' %(i))
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters, parameters_conv, grads, conv_grads

def FC_layer_model(X, Y, layers_dims, parameters = {}, batch_size = 64, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
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
        parameters = initialize_parameters_deep(layers_dims)
    m = X.shape[1]
    num_batchs = m // batch_size
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        for j in range(num_batchs):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            # AL is the output and caches contains Z, A, W, b for each layer
            AL, caches = L_model_forward(X[:, j*batch_size:(j+1)*batch_size], parameters)
            # Compute cost.
            cost = compute_cost(AL,Y[:, j*batch_size:(j+1)*batch_size])
            # Backward propagation.
            grads = L_model_backward(AL, Y[:, j*batch_size:(j+1)*batch_size], caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
            # Print the cost every 30 training example
            if print_cost and j % 300 == 0:
                print ("Cost after iteration %i, batch %i: %f" %(i, j, cost))
            if print_cost and j % 30 == 0:
                costs.append(cost)
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
#         if print_cost and i % 5 == 0:
#             costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters