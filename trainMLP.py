####################################################################################
# trainMLP.py takes a file containing training data as input and produces as output:
# - Five (5) files containing the trained neural network weights after 0 (for initial
#   weights), 10, 100, 1000, and 10,000 epochs.1 Use batch training, repeatedly
#   going over the training samples in-order, updating weights after each training
#   sample is run.
# - An image containing a plot of the learning curve. The learning curve represents
# the total sum of squared error (SSE) over all training samples after each epoch
#  (i.e. one complete pass over all training samples). Use the python matplotlib
# library (see http://matplotlib.org/users/index.html) to produce the plots.
####################################################################################
import math
import sys
import numpy as np
class MLP:
    def __init__(self,filename):
        f = open(filename,'r');







