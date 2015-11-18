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
# Import MLP class to make it available within the namespace of top-level module
from trainMLP import MLP
class executeMLP:
    def __init__(self, config_file, test_file):
        f = open(config_file,'rb')
        self.MLP = pickle.load(f)
        f.close()

        print(test_file)
        self.test_samples = self.parse_csv(test_file);
        print(self.test_samples)
    def parse_csv(self,filename):
        samples = [[float(i) for i in line.rstrip('\r\n').split(',')] for line in open(filename,'r')]
        return samples

CONFIG_FILE = '10_config'
TEST_FILE = 'test_data.csv'
exe = executeMLP(CONFIG_FILE, TEST_FILE);






