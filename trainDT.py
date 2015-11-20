import csv
import sys
import statistics
import math
import pickle
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
####################################################################################
# Node class
####################################################################################
class Node:

    def __init__(self):
        self.attribute=-1
        self.threshold=-1
        self.value=-1
        self.right=None
        self.left=None

####################################################################################
# getNodeCount : Returns the count of nodes
####################################################################################
    def getNodeCount(self):
        if self.left==None and self.right==None:
            return 1
        else:
            return self.left.getNodeCount()+self.right.getNodeCount()+1
####################################################################################
# getLeafCount : Returns the leaf count
####################################################################################
    def getLeafCount(self):
        if self.left==None and self.right==None:
            return 1
        else:
            return self.left.getLeafCount()+self.right.getLeafCount()

####################################################################################
# listPathLength : Returns the list of path lengths
####################################################################################
    def listPathLength(self,count,list):
        if self.left==None and self.right==None:
            list.append(count)
        else:
            count +=1
            self.left.listPathLength(count,list)
            self.right.listPathLength(count,list)

####################################################################################
# printTree : Prints the tree
####################################################################################
    def printTree(self,level,op):
        op.append([self.attribute,self.threshold,self.value,level])
        if self.left!=None:
            self.left.printTree(level+1,op)
        if self.right!=None:
            self.right.printTree(level+1,op)

####################################################################################
# printStatistics : Prints the Statistics for the tree
####################################################################################
    def printStatistics(self,name):
        path_list=[]
        self.listPathLength(0,path_list)
        path_list.sort()
        print("Summary for ",name," :")
        numOfNodes = self.getNodeCount()
        leafNodes=self.getLeafCount()
        print("Number of Nodes : ",numOfNodes)
        print("Number of Internal Nodes : ",numOfNodes-leafNodes)
        print("Number of Leaf Nodes : ",leafNodes)
        print("Max depth of root-to-leaf path : ",path_list[len(path_list)-1])
        print("Min depth of root-to-leaf path : ",path_list[0])
        print("Average depth of root-to-leaf path : ",statistics.mean(path_list))
        print()

####################################################################################
# export_configuration : Exports the file configuration
####################################################################################
    def export_configuration(self,filename):
        f = open(str(filename)+'_config','wb')
        pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        f.close()

###################################################################################
# plotSpaceDivision : Plots the decision tree space
####################################################################################
    def plotPoints(self,list,classified_values,name):
        op=[]
        x=np.arange(0,1.01,0.02)
        simulated_data=[]
        for i in range(0,len(x)):
            for j in range(0,len(x)):
                simulated_data.append([x[i],x[j]])
        op=self.classify_data(simulated_data)
        self.plotSpace(list,simulated_data,op,classified_values,name)

###################################################################################
# plotSpaceDivision : Plots the decision tree space
####################################################################################
    def plotSpaceDivision(self,list,name):
        op=[]
        space="Space"+name
        self.getPlotList(list,0,op)
        self.plotDivision(list,op,space)

####################################################################################
# getPlotList : Gets the list of co-ordinates for region split
####################################################################################
    def getPlotList(self,input_list,level,op):
        #check for leaf node
        if self.left != None and self.right != None:
            #split the data
            separated_data=split_data(input_list,self.attribute,self.threshold)
            left_data=separated_data[0]
            right_data=separated_data[1]
            #Splitting on attribute 1
            if self.attribute==0:
                x=self.threshold
                ymin=float(min(min(left_data,key=lambda x : x[1])[1],min(right_data,key=lambda x : x[1])[1]))
                ymax=float(max(max(left_data,key=lambda x : x[1])[1],max(right_data,key=lambda x : x[1])[1]))
                height=ymax-ymin
                xmin=float(min(min(left_data,key=lambda x : x[0])[0],min(right_data,key=lambda x : x[0])[0]))
                xmax=float(max(max(left_data,key=lambda x : x[0])[0],max(right_data,key=lambda x : x[0])[0]))
                left_width=xmin-x
                right_width=xmax-x
                op.append([x,ymin,left_width,height,level])
                op.append([x,ymin,right_width,height,level])
            else: #Splitting on attribute 2
                y=self.threshold
                xmin=float(min(min(left_data,key=lambda x : x[0])[0],min(right_data,key=lambda x : x[0])[0]))
                xmax=float(max(max(left_data,key=lambda x : x[0])[0],max(right_data,key=lambda x : x[0])[0]))
                width=xmax-xmin
                ymin=float(min(min(left_data,key=lambda x : x[1])[1],min(right_data,key=lambda x : x[1])[1]))
                ymax=float(max(max(left_data,key=lambda x : x[1])[1],max(right_data,key=lambda x : x[1])[1]))
                lower_height=ymin-y
                upper_height=ymax-y
                op.append([xmin,y,width,lower_height,level])
                op.append([xmin,y,width,upper_height,level])
            self.left.getPlotList(left_data,(level+1),op)
            self.right.getPlotList(right_data,(level+1),op)

####################################################################################
# getPlotList : Gets the list of co-ordinates for region split
####################################################################################
    def plotDivision(self,list,op,name):
        plot_list=[]
        color=['r','g','b','violet','gray','plum','m','brown','orange','purple']
        object_color=["red","lime","cyan","black"]
        fig= plt.figure()
        ax = fig.add_subplot(111)
        # Adding the rectangular patches
        for p in op:
            ax.add_patch(patches.Rectangle((float(p[0]),float(p[1])),float(p[2]),float(p[3]),fill=False,edgecolor=color[int(p[4])]))
        #Adding the scatter points to tree
        for p in range(len(list)):
            ax.scatter(float(list[p][0]),float(list[p][1]),c=object_color[int(list[p][2])-1])
        red_patch = patches.Patch(color='red', label='Bolt')
        lime_patch = patches.Patch(color='lime', label='Nut')
        cyan_patch = patches.Patch(color='cyan', label='Ring')
        black_patch = patches.Patch(color='black', label='Scrap')
        plt.legend(handles=[red_patch,lime_patch,cyan_patch,black_patch],fontsize = 'x-small')
        plt.ylabel('Rotational Symmetry')
        plt.xlabel('Eccentricity')
        plt.savefig(name)
        plt.show()

####################################################################################
# plotSpace : Plots the test points in space
####################################################################################
    def plotSpace(self,list,simulated_data,op,classified_values,name):
        fig= plt.figure()
        ax = fig.add_subplot(111)
        region_color = ['#FFEFD5','#87CEEB','#EE82EE','#B0C4DE']
        object_color=['red','blue','green','yellow']
        for p in range(len(simulated_data)):
            ax.scatter(float(simulated_data[p][0]),float(simulated_data[p][1]),c=object_color[int(op[p])-1],s=80,alpha=0.2)
        for p in range(len(list)):
            ax.scatter(float(list[p][0]),float(list[p][1]),c=object_color[int(classified_values[p])-1],s=120,alpha=1)
        red_patch = patches.Patch(color='red', label='Bolt')
        lime_patch = patches.Patch(color='blue', label='Nut')
        cyan_patch = patches.Patch(color='green', label='Ring')
        black_patch = patches.Patch(color='yellow', label='Scrap')
        plt.legend(handles=[red_patch,lime_patch,cyan_patch,black_patch],fontsize = 'x-small')
        plt.ylabel('Rotational Symmetry')
        plt.xlabel('Eccentricity')
        plt.savefig(name)
        plt.show()


####################################################################################
# classify_data : Classifies the test data points
####################################################################################
    def classify_data(self,list):
        class_value=[]
        for i in range(0,len(list)):
            element=list[i]
            class_value.append(self.get_value(element))
        return class_value

####################################################################################
# get_value : Gets the value for classified points
####################################################################################
    def get_value(self,element):
        if self.left==None and self.right==None:
            return self.value
        if float(element[self.attribute])< float(self.threshold):
            return self.left.get_value(element)
        else:
            return self.right.get_value(element)

####################################################################################
# getEntropy : Returns the entropy for the points
####################################################################################
def getEntropy(target_values,total):
    result=0
    for key in target_values.keys():
        result += -(target_values[key]/total)*np.log2((target_values[key]/total))
    return result

####################################################################################
# addData : Adds the data to the dict
####################################################################################
def addData(data,key):
    if key in data:
        data[key] += 1
    else:
        data[key] = 1

####################################################################################
# get_information_gain : Gets the information gain for the attribute
####################################################################################
def get_information_gain(input_data,attribute,target_values):
    threshold_values = []
    attr_values=[]
    # Gets the attribute value and sort
    for i in range(0, len(input_data)):
        attr_values.append(float(input_data[i][attribute]))
    attr_values.sort()
    # Create the thresholds for the information gain
    if len(attr_values) <= 1:
        threshold_values.append(attr_values[0])
    else:
        for i in range(0, len(input_data)-1):
            threshold_values.append((attr_values[i]+attr_values[i+1])/2)
    threshold_values.sort()
    chk_parameter = len(input_data[0])-1
    entropy_list = []
    # Loop for all the threshold values:
    for i in range(0, len(threshold_values)):
        threshold_start = float(threshold_values[i])
        left_data=dict()
        right_data=dict()
        i = 0
        total_left = 0
        total_right = 0
        #for all the data values calculate the calculating the class type on each side of threshold
        while i < len(input_data):
            temp_list = input_data[i]
            if float(temp_list[attribute]) < threshold_start:
                total_left += 1
                addData(left_data,temp_list[chk_parameter])
            else:
                total_right += 1
                addData(right_data,temp_list[chk_parameter])
            i = i + 1

        total = total_right + total_left
        # Calculates the information gain for the parent
        information_gain = getEntropy(target_values,total)
        if total_left != 0:
            #print((total_left/total)*getEntropy(left_data,total_left),"left")
            information_gain -=(total_left/total)*getEntropy(left_data,total_left)
        if total_right != 0:
            #print((total_right/total)*getEntropy(right_data,total_right),"right")
            information_gain -=(total_right/total)*getEntropy(right_data,total_right)
        #print(information_gain)
        entropy_list.append([attribute,threshold_start,information_gain])
    entropy_list = max(entropy_list, key=lambda x: float(x[2]))
    # Returns the information for max information gain
    return entropy_list

####################################################################################
# get_best_attribute : Returns the data for best attribute to split
####################################################################################
def get_best_attribute(input_data,attribute_list,target_values):
    result=[]
    for each in attribute_list:
        result.append(get_information_gain(input_data,each,target_values))
    result = sorted(result, key=lambda x: (x[2]))
    return result[len(result)-1]

####################################################################################
# split_data : Partitions the data in 2 separate list based on threshold value
####################################################################################
def split_data(input_data,attribute,threshold):
    left_list=[]
    right_list=[]
    left_values=dict()
    right_values=dict()
    chk_parameter = len(input_data[0])-1
    # iterate over the list
    for item in input_data:
        if float(item[attribute])<threshold:
            left_list.append(item)
            addData(left_values,item[chk_parameter])
        else:
            right_list.append(item)
            addData(right_values,item[chk_parameter])
    #returns the partitioned list
    return [left_list,right_list,left_values,right_values]

####################################################################################
# getLeaf : Returns the leaf node
####################################################################################
def getLeaf(attribute,val):
    node = Node()
    node.attribute=attribute
    node.value=val
    return node

####################################################################################
# decision_tree : Generates and returns the decision tree
####################################################################################
def decision_tree(input_data,attribute_list,class_values):
    if len(input_data)>0:
        root=Node()
        #Get the best attribute to split the data
        split_attribute=get_best_attribute(input_data,attribute_list,class_values)
        root.attribute=split_attribute[0]
        root.threshold=split_attribute[1]
        information_gain=split_attribute[2]
        max=0
        counter = 0
        #Get the max value for the root
        for key in class_values.keys():
            if class_values[key]>counter:
                    max=key
                    counter=class_values[key]
        root.value=max
        # Return if all the values belong to same class
        if len(class_values) == 1:
            return root
        # Partition the data
        separated_data = split_data(input_data,root.attribute,root.threshold)
        left_data = separated_data[0]
        right_data = separated_data[1]
        left_values = separated_data[2]
        right_values = separated_data[3]
        # Check if tree consists of both child
        if len(left_data)!=0 and len(right_data)!=0:
            root.left=decision_tree(left_data,attribute_list,left_values)
            root.right=decision_tree(right_data,attribute_list,right_values)
        elif len(left_data)==0:
            #Checks if the information gains is 0
            if information_gain!=0:
                root.right=decision_tree(right_data,attribute_list,right_values)
            else:
                root.right=getLeaf(root.attribute,root.value)
            root.left=getLeaf(root.attribute,root.value)
        else:
            #Checks if the information gains is 0
            if information_gain!=0:
                root.left=decision_tree(left_data,attribute_list,left_values)
            else:
                root.left=getLeaf(root.attribute,root.value)
            root.right=getLeaf(root.attribute,root.value)
        return root

    else:
        return None

####################################################################################
# chiSquarePruning : Prunes the tree for the given significance level
####################################################################################
def chiSquarePruning(tree,input_list,threshold):
    # Check for the root node
    if tree.left == None and tree.right == None:
        return True
    else:
        # Partition the data
        separated_data=split_data(input_list,tree.attribute,tree.threshold)
        # Check if the both the child are pruned
        if chiSquarePruning(tree.left,separated_data[0],threshold) and chiSquarePruning(tree.right,separated_data[1],threshold):
            # Generates the class values for the parent and child
            class_values=dict()
            left_values=separated_data[2]
            right_values=separated_data[3]
            op=len(input_list[0])-1
            for row in input_list:
                addData(class_values,row[op])

            # calculate the degree of freedom for the parent
            freedom=len(class_values)-1
            total_left=sum(left_values.values())
            total_right=sum(right_values.values())
            left_chisquare=0
            right_chisquare=0
            # Generate the chisquare value for the left child
            if total_left != 0:
                for key in left_values:
                    expected_value= (total_left/(total_left+total_right))*class_values[key]
                    left_chisquare += math.pow((left_values[key]-expected_value),2)/expected_value
            # Generate the chisquare value for the right child
            if total_right != 0:
                for key in right_values:
                    expected_value= (total_right/(total_left+total_right))*class_values[key]
                    right_chisquare += math.pow((right_values[key]-expected_value),2)/expected_value

            # Check for pruning the tree
            if (left_chisquare+right_chisquare)>threshold[freedom]:
                max=0
                counter = 0
                for key in class_values.keys():
                    if class_values[key]>counter:
                        max=key
                        counter=class_values[key]
                tree.value=max
                tree.left=None
                tree.right=None
                return True

        return False

####################################################################################
# parse_csv : Reads the csv file
####################################################################################
def parse_csv(filename):
        with open(filename) as f:
            lines = filter(None,(line.rstrip() for line in f))
            samples = [[i for i in line.split(',')] for line in lines]
        return samples

####################################################################################
# main()
####################################################################################
def main():
    if len(sys.argv)<2:
        print('Please enter input file name, for instance: python3 trainDT.py train_data.csv')
    else:
        attributes=[]
        filename=sys.argv[1]
        significance = {3:7.815,2:5.991,1:3.841}
        # Extracting the Training data
        #list,class_values=readFile(filename)
        list=parse_csv(filename)
        class_values=dict()
        output_index=len(list[0])-1
        for x in list:
            addData(class_values,x[output_index])
        for x in range(1,len(list[0])):
            attributes +=[x-1]
        tree=decision_tree(list,attributes,class_values)
        tree.printStatistics("Decision Tree")
        tree.export_configuration("decisionTree")
        tree.plotSpaceDivision(list,"decisionTree.png")
        chiSquarePruning(tree,list,significance)
        tree.printStatistics("Pruned Tree")
        tree.export_configuration("prunedTree")
        tree.plotSpaceDivision(list,"PrunedTree.png")
if __name__ == "__main__":
    main()