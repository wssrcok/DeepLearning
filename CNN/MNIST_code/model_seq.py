from layers.cnn_layers import *
from layers.fc_layers import *

def conv2d_f(hparameters = {'stride': 1, 'pad': 2}, truncate = 0):
    def layer(A_prev, W, b):
        n_C, n_C_prev, f, f = W.shape
        m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape

        stride = hparameters['stride']
        pad = hparameters['pad']
        A_prev_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
        W_col = W.reshape(n_C, -1)

        out = W_col @ A_prev_col + b
        n_H_shape = (n_H_prev - f + 2 * pad) / stride + 1
        n_W_shape = (n_W_prev - f + 2 * pad) / stride + 1
        n_H_shape, n_W_shape = int(n_H_shape), int(n_W_shape)

        out = out.reshape(n_C, n_H_shape, n_W_shape, m)
        out = out.transpose(3, 0, 1, 2)
        # now Z.shape is (m, n_C, n_H_shape, n_W_shape)

        cache = (A_prev, W, b, hparameters, A_prev_col)
        return out, cache
    return layer

def Quantized_ReLu():
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    def layer(Z, truncate = False):
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        if truncate:
            A = truncate_io(A)
        return A, cache
    return layer

def Sequential_f(moduleList):
    def model(x, parameters):
        output = x
        for i,m in enumerate(moduleList):
            output, cache = m(output, parameters)
        return output, cache
    return model

def cnn_model_general(input_feature, label, filter_dims, layers_dims, truncate = 0, parameters = {}, parameters_conv = {}, batch_size = 64, learning_rate = 0.01, num_iterations = 100, print_cost=False):
    """
    Arguments:
    
    Returns:
    """
    costs = []             # keep track of cost
    
    # Parameters initialization
    if len(parameters) == 0:
        parameters_conv = initialize_parameters_filter_general(filter_dims, truncate = truncate)
        parameters = initialize_parameters_deep(layers_dims, truncate = truncate)

    m = input_feature.shape[0]                    # m: total number of training dataset
    num_batchs = m // batch_size
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # right now batch is not randomized
        # TODO: randomize batch every epoch.
        for j in range(num_batchs):
            # conv forward
            input_batch = input_feature[j*batch_size:(j+1)*batch_size]
            cout, conv_caches = conv_pool_forward_general(input_batch, parameters_conv, truncate = truncate)
            #flatten
            (a,b,c,d) = cout.shape
            cout = cout.reshape(batch_size,-1).T
            # fc forward
            fc_out, caches = L_model_forward(cout, parameters, truncate = truncate)
            # prevent divide by zero occur.
            if truncate:
                fc_out = np.where(fc_out == 0,1/256,fc_out)
                fc_out = np.where(fc_out == 1,255/256,fc_out)
            # Compute cost.
            cost = compute_cost(fc_out,label[:, j*batch_size:(j+1)*batch_size])
            # fc backward
            grads = L_model_backward(fc_out, label[:, j*batch_size:(j+1)*batch_size], caches)
            # unflatten
            dcout = grads['dA0'].T
            dcout = dcout.reshape(a,b,c,d)
            # conv backward
            conv_grads = conv_pool_backward_general(dcout, conv_caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate, truncate = truncate)
            parameters_conv = update_conv_parameters_general(parameters_conv, conv_grads, learning_rate, truncate = truncate)
    
    return parameters, parameters_conv, grads, conv_grads

model_f = Sequential_f([
    conv2d_f(),
    relu(),
    Quantized_ReLu(),
    conv2d_f()
])
