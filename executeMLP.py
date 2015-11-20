####################################################################################
# executeMLP.py takes a file defining a trained network and a data file, and runs
# the network on the data file. The program should produce:
# 1.(standard output) On the command line:
# i. Recognition rate (% correct)
# ii. Profit obtained
# iii. A confusion matrix, which is a histogram counting the number of occurrences
# for each combination of assigned class (rows) and actual class (columns). The main
# diagonal contains counts for correctly assigned samples, all remaining cells
# correspond to counts for di erent types of error.
# 2.(image file) A plot of the test samples (using shapes/color to represent classes)
# along with the classification regions. The program runs the current network over
# a grid of equally spaced points in the region (0,0) to (1,1) (the limits of the
# feature space), using a di erent color to represent the assigned class (i.e
# . classification decision by the MLP) at each point.
####################################################################################
import pickle
import numpy as np
# Import MLP class to make it available within the namespace of top-level module
from trainMLP import MLP
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
####################################################################################
# Define profits matrix
####################################################################################
PROFIT = [[20,-7,-7,-7],
           [-7,15,-7,-7],
           [-7,-7,5,-7],
           [-3,-3,-3,-3]];
####################################################################################
# Initialize matrix, not sure we can use python for this
####################################################################################
def initialize_matrix(m,n,fill = 0.0):
    matrix = []
    for i in range(m):
        matrix.append([fill]*n)
    return matrix;
class executeMLP:
    def __init__(self, config_file, test_file):
        f = open(config_file,'rb')
        self.MLP = pickle.load(f)
        f.close()
        self.test_samples = self.MLP.parse_csv(test_file);
        self.vector_target = [int(row[2]) for row in self.test_samples];
        #print(self.vector_target)
    def runNN(self):
        self.test_result = []
        self.vector_result = []
        for t in self.test_samples:
            self.test_result.append(self.MLP.feedForward(t).copy())
        for res in self.test_result:
            r = [l for l in res if l >=0.5]
            if len(r)==1:
                self.vector_result.append(res.index(r)+1)
            else:
                self.vector_result.append(4)
        #print(self.vector_result)
        self.getoutput()
    def getoutput(self):
        correct = 0
        #print(self.vector_target)
        len_sample = len(self.test_samples)
        output_profit = 0;
        confusion_matrix = initialize_matrix(4,4);
        for i in range(len_sample):
            if self.vector_result[i] == self.vector_target[i]:
                correct+=1
            output_profit += PROFIT[self.vector_result[i]-1][self.vector_target[i]-1]
            confusion_matrix[self.vector_result[i]-1][self.vector_target[i]-1] +=1;
        recognition_rate = correct/len_sample*100
        print('Recognition rate: %.0f%%'%(recognition_rate));
        print('Profit obtained: %.0f'%(output_profit))
        print('Confusion matrix:\n')
        # print(confusion_matrix)
        for row in confusion_matrix:
            print(row)

        self.plotdata(self.test_samples,self.vector_result)

    def plotdata(self,list,classified_values):
        fig= plt.figure()
        ax = fig.add_subplot(111)
        object_color=["red","lime","cyan","blue"]
        for i in range(len(list)):
                ax.scatter(float(list[i][0]),float(list[i][1]),c=object_color[int(classified_values[i])-1])
        red_patch = patches.Patch(color='red', label='Bolt')
        lime_patch = patches.Patch(color='lime', label='Nut')
        cyan_patch = patches.Patch(color='cyan', label='Ring')
        black_patch = patches.Patch(color='blue', label='Scrap')
        plt.legend(handles=[red_patch,lime_patch,cyan_patch,black_patch],fontsize = 'x-small')
        plt.ylabel('Rotational Symmetry')
        plt.xlabel('Eccentricity')
        plt.savefig('MLPTestData.png')
        plt.show()

CONFIG_FILE = '1000_config'
TEST_FILE = 'test_data.csv'
exe = executeMLP(CONFIG_FILE, TEST_FILE);
exe.runNN()




