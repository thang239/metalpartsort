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
import csv
import pickle
import numpy as np
# Import MLP class to make it available within the namespace of top-level module
from trainDT import Node
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
def initialize_matrix(m,n,fill = 0):
    matrix = []
    for i in range(m):
        matrix.append([fill]*n)
    return matrix;

def compare_result(list,results):
    count=0
    for i in range(0,len(list)):
        if (list[i])==(results[i]):
            count +=1
    return float(count/len(list))


def classify_data(list,classify_tree):
    class_value=[]
    for i in range(0,len(list)):
        element=list[i]
        class_value.append(get_value(element,classify_tree))
    return class_value

def get_value(element,root):
    if root.left==None and root.right==None:
            return root.value
    if float(element[root.attribute])< float(root.threshold):
            return get_value(element,root.left)
    else:
            return get_value(element,root.right)

def readFile(filename):
    class_values=dict()
    list = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            list += [row]
            output_index=len(row)-1
            if row[output_index] in class_values:
                class_values[row[output_index]] +=1
            else:
                class_values[row[output_index]] =1

    return list,class_values

def parse_csv(filename):
        with open(filename) as f:
            lines = filter(None,(line.rstrip() for line in f))
            samples = [[i for i in line.split(',')] for line in lines]
        return samples

def testStatistics(list,tree):
    op=len(list[0])-1
    classified_values= classify_data(list,tree)
    #recognition_rate=compare_result(original_output,classified_values)
    confusion_matrix = initialize_matrix(4,4)
    count=0
    profit=0
    for i in range(len(list)):
        element=int(list[i][op])
        predicted_val=int(classified_values[i])
        if element == predicted_val:
           count +=1
        profit += PROFIT[predicted_val-1][element-1]
        confusion_matrix[predicted_val-1][element-1] += 1;
    recognition_rate = count/len(list)*100

    print('Recognition rate: %.0f%%'%(recognition_rate));
    print('Profit obtained: %.0f'%(profit))
    print('Confusion matrix:')
        # print(confusion_matrix)
    for row in confusion_matrix:
        print(row)
######################################
#  main()
######################################
def main():
    CONFIG_FILE = 'prunedTree_config'
    TEST_FILE = 'test_data.csv'

    f = open(CONFIG_FILE,'rb')
    tree = pickle.load(f)
    f.close()
    list=parse_csv(TEST_FILE)
    testStatistics(list,tree)
if __name__ == "__main__":
    main()





