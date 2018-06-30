# DeepLearning
CNN model using numpy
# Useage
example: MNIST
go to file directory: /Deeplearning/CNN/MNIST_code
run $ python main.py
the program will starts to print cost at each level
# Limitation
right now, only have demo for MNIST
the default is layers are: CONV->RELU->POOL->CONV->RELU->POOL->FLATTEN->FC->FC->SOFTMAX->OUTPUT
Layers are easy to change: in main.py, change filters_dim and layers_dim. 
Padding and Stride are harder to change: in MNIST_code/layers/conv_layers.py, function conv_forward and function pool forward, change the hyperparameter to a desired value.
