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
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Import MLP class to make it available within the namespace of top-level module
from trainMLP import MLP

####################################################################################
# Define profits matrix
####################################################################################
PROFIT = [[20,-7,-7,-7],
           [-7,15,-7,-7],
           [-7,-7,5,-7],
           [-3,-3,-3,-3]];

MATCH = {
    1:"Bolts",
    2:"Nuts",
    3:"Rings",
    4:"Scrap"
}
color = ['ro','bo','go','yo']
region_color = ['r','b','g','y']
# region_color = ['#FFEFD5','#87CEEB','#EE82EE','#B0C4DE']
# region_color = [[204, 255, 204],[255, 204, 255],[255, 204, 204],[204, 255, 255]]
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
        self.config_file = config_file
        f = open(self.config_file,'rb')
        self.MLP = pickle.load(f)
        f.close()

        self.test_samples = self.MLP.parse_csv(test_file);
        self.vector_target = [int(row[2]) for row in self.test_samples];
        # print(self.vector_target)
    def runNN(self):
        self.test_result = []
        self.vector_result = []
        for t in self.test_samples:
            self.test_result.append(self.MLP.feedForward(t).copy())
        for res in self.test_result:
            self.vector_result.append(self.classify(res))
        self.getoutput()
        # self.plot_image()
    def classify(self, res):
        r = [l for l in res if l >=0.5]
        if len(r)==1:
            return res.index(r)+1
        else:
            m = max(res)
            if m > 0.3:
                return res.index(m)+1
            else:
                return 1
    def getoutput(self):
        correct = 0
        len_sample = len(self.test_samples)
        output_profit = 0;
        confusion_matrix = initialize_matrix(4,4);
        for i in range(len_sample):
            if self.vector_result[i] == self.vector_target[i]:
                correct+=1
            output_profit += PROFIT[self.vector_result[i]-1][self.vector_target[i]-1]
            confusion_matrix[self.vector_result[i]-1][self.vector_target[i]-1] +=1;
        recognition_rate = correct/len_sample*100
        print('Number of epochs: '+self.config_file.split('_')[0])
        print('Recognition rate: %.0f%%'%(recognition_rate));
        print('Profit obtained: %.0f'%(output_profit))
        print('Confusion matrix:\n')

        # for row in confusion_matrix:
        #     print(row,'\n')
        names=["Bolts"," Nuts","Rings","Scrap"]
        print("     Bolts   Nuts  Rings  Scrap")
        for row in range(len(confusion_matrix)):
            print(names[row],end="  ")
            element=confusion_matrix[row]
            for each in element:
                print(each,end="      ")
            print()
    def plot_image(self):

        separate = [np.asmatrix([sample for sample in self.test_samples if sample[2]==1]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==2]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==3]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==4])];
        for j in range(len(separate)):
            plt.plot(separate[j][:,0],separate[j][:,1],color[j],label=MATCH[j+1])
        plt.xlabel('Six-fold rotational symmetry')
        plt.ylabel('Eccentricity')

        plt.title('Testing set')
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.savefig('./figure/testing_set.png')
    def plot_classification_regions(self):
        step = 0.01
        lg = []
        art = []
        lb_regions = {
            0:'Bolts region',
            1:'Nuts region',
            2:'Rings region',
            3:'Scrap region'
        }
        lb = {
            0:'Bolts',
            1:'Nuts',
            2:'Rings',
            3:'Scrap'
        }
        for j in range(int(1/step)):
            j = j*step
            for k in range(int(1/step)):
                k = k *step
                sample = [j,k]
                # print(sample)
                recog = self.classify(self.MLP.feedForward(sample).copy())
                plt.plot(j,k,color = region_color[recog-1], marker = 'o',markeredgecolor = 'none',alpha = 0.4)
        separate = [np.asmatrix([sample for sample in self.test_samples if sample[2]==1]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==2]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==3]),
                    np.asmatrix([sample for sample in self.test_samples if sample[2]==4])];
        for j in range(len(separate)):
            plt.plot(separate[j][:,0],separate[j][:,1],color[j])

        plt.xlabel('Six-fold rotational symmetry')
        plt.ylabel('Eccentricity')

        plt.title('Classification regions for %s training epochs'%(self.config_file.split('_')[0]))
        for i in range(4):
            # lg.append(mpatches.Patch(color=region_color[i], label=lb_regions[i]))
            lg.append(mpatches.Patch(color=color[i][0], label=lb[i]))
        lgd = plt.legend(handles=lg,fontsize = 'x-small',loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
        art.append(lgd)
        plt.savefig('./figure/'+self.config_file+'_classification_region_MLP.png',additional_artists=art,
    bbox_inches="tight")

        # plt.show()
        plt.clf()
CONFIG_FILE = ['0_config','10_config','100_config','1000_config','10000_config']
TEST_FILE = 'test_data.csv'
# TRAIN_FILE = 'train_data.csv'

def main():
    if len(sys.argv)==2:
        exe = executeMLP(sys.argv[1], TEST_FILE);
        exe.runNN()
        exe.plot_classification_regions()
    else:
        for i in range(0,5):
            exe = executeMLP(CONFIG_FILE[i], TEST_FILE);
            exe.runNN()
            # exe.plot_classification_regions()
    # exe.plot_image()
if __name__ == "__main__":
    main()


