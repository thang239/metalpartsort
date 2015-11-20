####################################################################################
# train and execute MLPs with one hidden layer
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
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

####################################################################################
# Define sigmoid function and its derivation
####################################################################################
def sigmoid(x):
    return 1/(1+np.exp(-x));
def derivative_sigmoid(a):
    return a * (1-a);
def match_output(num):
    match = {
        1.0:[1,0,0,0],
        2:[0,1,0,0],
        3:[0,0,1,0],
        4:[0,0,0,1],
    }
    return match[num]
####################################################################################
# MLP class, contain main function for neural network
####################################################################################
class MLP:
    def __init__(self, filename, layers, input_node, hidden_node, output_node):
        self.samples = self.parse_csv(filename)
        # Setup parameters
        # self.epochs = epochs;
        self.input = input_node + 1;
        self.hidden_node = hidden_node;
        self.output_node = output_node;
        self.L = layers

        # Setup activations
        self.ai = [1.0]*self.input;
        self.ah = [1.0]*self.hidden_node;
        self.ao = [1.0]*self.output_node;

        # Initialize weights for all node, using np.random.rand for convenience
        self.weight_ih = 2 * np.random.rand(self.input, self.hidden_node)-1;
        self.weight_ho = 2 * np.random.rand(self.hidden_node, self.output_node)-1;

        #Store error for each epochs
        self.error = []
    def feedForward(self, sample):
        for i in range(self.input-1):
            self.ai[i+1] = sample[i]
        # print(self.ai)
        # Update activations in hidden node
        for i in range(self.hidden_node):
            temp = 0.0
            for j in range(self.input):
                temp+=self.ai[j]*self.weight_ih[j][i]
            self.ah[i] = sigmoid(temp);
        # Update activations in hidden node
        for i in range(self.output_node):
            temp = 0.0
            for j in range(self.hidden_node):
                temp+=self.ah[j]*self.weight_ho[j][i]
            self.ao[i] = sigmoid(temp);
        # print(self.ao)
        return self.ao
    def backProgapate(self, sample, learning_rate):
        delta_output = [0.0]*self.output_node;
        delta_hidden = [0.0]*self.hidden_node;

        output = match_output(sample[-1]);
        #Compute delta output
        for j in range(self.output_node):
            delta_output[j] = derivative_sigmoid(self.ao[j])*(output[j]-self.ao[j])
        #Compute delta hidden
        for j in range(self.hidden_node):
            temp = 0.0
            for k in range(self.output_node):
                temp+=self.weight_ho[j][k]*delta_output[k]
            delta_hidden[j] = derivative_sigmoid(self.ah[j])*temp;

        # Update the weights from hidden to output
        for i in range(self.hidden_node):
            for j in range(self.output_node):
                self.weight_ho[i][j] = self.weight_ho[i][j] + learning_rate * self.ah[i]*delta_output[j];
        # Update the weights from input to hidden
        for i in range(self.input):
            for j in range(self.hidden_node):
                self.weight_ih[i][j] = self.weight_ih[i][j] + learning_rate * self.ai[i]*delta_hidden[j];
        #Calculate error for current epoch
        error = 0.0
        for i in range(self.output_node):
            error+=1/2*(output[i]-self.ao[i])**2;
        return error

    def train(self, learning_rate, iterations):
        # print(self.weight_ho);
        # print(self.weight_ih);
        print('I am training ...')
        dump_epochs = [0,10,100,1000,10000];
        len_samples = len(self.samples)
        sse = []
        for i in range(iterations+1):
            errList = []
            for sample in self.samples:
                # print(sample)
                self.feedForward(sample)
                errList.append(self.backProgapate(sample, learning_rate))
            if i in dump_epochs:
                self.export_configuration(i)
            sse.append(sum(errList)/len_samples);
        self.dump_sse_and_plot_learning_curve(sse)
    def parse_csv(self,filename):
        with open(filename) as f:
            lines = filter(None,(line.rstrip() for line in f))
            samples = [[float(i) for i in line.split(',')] for line in lines]
        return samples
    def export_configuration(self,filename):
        f = open(str(filename)+'_config','wb')
        pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        f.close()
    def dump_sse_and_plot_learning_curve(self,sse):
        f = open('error.csv','w')
        xvalues = []
        for s in sse:
            f.write('%s\n'%s)
            xvalues.append(sse.index(s))
        # print(xvalues)
        plt.plot(xvalues,sse,'r')
        plt.xlabel('Epochs')
        plt.ylabel('Sum squared error')
        plt.title('Sum squared error after each epoch')
        plt.savefig('learning_curve.png')
<<<<<<< HEAD
        plt.show()
=======
>>>>>>> ffff3eb1ed4c4ae3c9c5f89b28cb7771a31bc970
    # def plot_learning_curve(self):


####################################################################################
# Define parameters
####################################################################################
NUM_LAYERS = 3
NUM_INPUT_NODE = 2
NUM_HIDDEN_NODE = 5
NUM_OUTPUT_NODE = 4
NUM_EPOCHS = 10000
LEARNING_RATE = 0.1
def main():
    if len(sys.argv)<2:
        print('Please enter input file name, for instance: python3 trainLMP.py train_data.csv 10000')
    else:
        nnet = MLP(sys.argv[1], NUM_LAYERS, NUM_INPUT_NODE, NUM_HIDDEN_NODE, NUM_OUTPUT_NODE)
        if sys.argv[2]==None:
            nnet.train(LEARNING_RATE, NUM_EPOCHS)
        else:
            nnet.train(LEARNING_RATE, int(sys.argv[2]))
if __name__ == "__main__":
    main()
