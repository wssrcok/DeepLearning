import time
import math
import sys
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
from scipy import ndimage
from utils import load_dataset
from model import cnn_model_general, cnn_model

np.random.seed(2)

def main():
	train_data, train_labels, eval_data, eval_labels, classes = load_dataset();

	# two conv layers and 2 fc layers.
	filter_dims = [(32,1,5,5),(64,32,5,5)] 
	fc_dims = [3136, 1024, classes]
	parameters_fc, parameters_conv, grads_fc, conv_grads = cnn_model_general(
		train_data[0:256], 
		train_labels[0:256], 
		filter_dims, 
		fc_dims, 
	    truncate = 0, 
	    batch_size = 32, 
	    learning_rate = 0.02, 
	    num_iterations = 10, 
	    print_cost = True)

if __name__ == '__main__':
	main()