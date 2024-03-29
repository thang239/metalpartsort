####################################################################################
# executeDT.py takes a file defining a trained network and a data file, and runs
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
import sys
import pickle
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

####################################################################################
# parse_csv : Reads the csv file
####################################################################################
def parse_csv(filename):
        with open(filename) as f:
            lines = filter(None,(line.rstrip() for line in f))
            samples = [[i for i in line.split(',')] for line in lines]
        return samples

####################################################################################
# plotdata
####################################################################################
def plotdata(list,classified_values,name):
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
    plt.savefig(name)
    plt.show()

####################################################################################
# testStatistics() :  Prints and plots the statistics for test data
####################################################################################
def testStatistics(list,tree,name):
    op=len(list[0])-1
    #Classifies the input values
    classified_values= tree.classify_data(list)
    #initialize the confusion matrix
    confusion_matrix = initialize_matrix(4,4)
    count=0
    profit=0

    #Calculate the profit and recognition rate
    for i in range(len(list)):
        element=int(list[i][op])
        predicted_val=int(classified_values[i])
        if element == predicted_val:
           count +=1
        profit += PROFIT[predicted_val-1][element-1]
        confusion_matrix[predicted_val-1][element-1] += 1;
    recognition_rate = count/len(list)*100

    print('Recognition rate: %.0f%%'%(recognition_rate));
    print('Profit obtained: %.2f$'%(profit/100))
    print('Confusion matrix:')
        # print(confusion_matrix)
    names=["Bolts"," Nuts","Rings","Scrap"]
    print("     Bolts   Nuts  Rings  Scrap")
    for row in range(len(confusion_matrix)):
        print(names[row],end="  ")
        element=confusion_matrix[row]
        for each in element:
            print(each,end="      ")
        print()
    # Plot the data
    tree.plotPoints(list,classified_values,name)

####################################################################################
#   loadObject() : Loads the object
####################################################################################
def loadObject(name):
    f = open(name,'rb')
    tree = pickle.load(f)
    f.close()
    return tree
####################################################################################
#   main()
####################################################################################
def main():
    if len(sys.argv)<2:
        print('Please enter input file name, for instance: python3 executeDT.py test_data.csv')
    else:
        TEST_FILE = sys.argv[1]
        list=parse_csv(TEST_FILE)
        print("Decision Tree Statistics :")
        print()
        tree=loadObject('decisionTree_config')
        testStatistics(list,tree,"DecisionTestData.png")
        tree=loadObject('prunedTree_config')
        print()
        print("Pruned Tree Statistics :")
        print()
        testStatistics(list,tree,"PrunedTestData.png")

if __name__ == "__main__":
    main()





