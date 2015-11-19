import csv
import statistics
import math
import pickle

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
######################################
# Node Class
#
######################################
class Node:

    def __init__(self):
        self.attribute=-1
        self.threshold=-1
        self.value=-1
        self.right=None
        self.left=None

    def getNodeCount(self):
        if self.left==None and self.right==None:
            return 1
        else:
            return self.left.getNodeCount()+self.right.getNodeCount()+1

    def getLeafCount(self):
        if self.left==None and self.right==None:
            return 1
        else:
            return self.left.getLeafCount()+self.right.getLeafCount()

    def listPathLength(self,count,list):
        if self.left==None and self.right==None:
            list.append(count)
        else:
            count +=1
            self.left.listPathLength(count,list)
            self.right.listPathLength(count,list)

    def printStatistics(self,name):
        path_list=[]
        self.listPathLength(0,path_list)
        path_list.sort()
        print("Summary for ",name," :")
        print("Number of Nodes : ",self.getNodeCount())
        print("Number of Leaf Nodes : ",self.getLeafCount())
        print("Max depth of root-to-leaf path : ",path_list[len(path_list)-1])
        print("Min depth of root-to-leaf path : ",path_list[0])
        print("Average depth of root-to-leaf path : ",statistics.mean(path_list))

    def export_configuration(self,filename):
        f = open(str(filename)+'_config','wb')
        pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        f.close()


######################################
#
#
######################################
def getEntropy(target_values,total):
    result=0
    for key in target_values.keys():
        result += -(target_values[key]/total)*math.log((target_values[key]/total),2)
    return result

######################################
#
#
######################################

def addData(data,key):
    if key in data:
        data[key] += 1
    else:
        data[key] = 1

######################################
#
#
######################################
def get_information_gain(input_data,attribute,target_values):
    threshold_values = []
    if len(input_data) <= 1:
        threshold_values.append(input_data[0][attribute])
    else:
        for i in range(0, len(input_data)-1):
            threshold_values.append((float(input_data[i][attribute])+float(input_data[i+1][attribute]))/2)
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
        #for all the data values calculate the calculating the number of 0's,1's on each side of threshold
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
        information_gain = getEntropy(target_values,total)
        if total_left != 0:
            information_gain -=(total_left/total)*getEntropy(left_data,total_left)
        if total_right != 0:
            information_gain -=(total_right/total)*getEntropy(right_data,total_right)
        entropy_list.append([attribute,threshold_start,information_gain])

    entropy_list = sorted(entropy_list, key=lambda x: float(x[2]))
    return entropy_list[int((len(entropy_list)-1)/2)]

######################################
#
#
######################################
def get_best_attribute(input_data,attribute_list,target_values):
    result=[]
    for each in attribute_list:
        #print(each,len(input_data),target_values,"inp len")
        result.append(get_information_gain(input_data,each,target_values))
    #print(result)
    result = sorted(result, key=lambda x: float(x[2]))
    return result[len(result)-1]

######################################
# separate_data : Partition the list
# according to the threshold
######################################
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
    #returns the partiton list
    return [left_list,right_list,left_values,right_values]

######################################
# get_max_key : Gets the key with max value
#
######################################
def getLeaf(attribute,val):
    node = Node()
    node.attribute=attribute
    node.value=val
    return node

######################################
#
#
######################################
def decision_tree(input_data,attribute_list,class_values):
    if len(input_data)>0:
        root=Node()
        split_attribute=get_best_attribute(input_data,attribute_list,class_values)
        root.attribute=split_attribute[0]
        root.threshold=split_attribute[1]
        information_gain=split_attribute[2]
        root.value=max(class_values)[0]
        if len(class_values) == 1:
            return root
        separated_data = split_data(input_data,root.attribute,root.threshold)
        left_data = separated_data[0]
        right_data = separated_data[1]
        left_values = separated_data[2]
        right_values = separated_data[3]
        if len(left_data)!=0 and len(right_data)!=0:
            root.left=decision_tree(left_data,attribute_list,left_values)
            root.right=decision_tree(right_data,attribute_list,right_values)
        elif len(left_data)==0:
            if information_gain!=0:#Checks if the information gains is 0
                root.right=decision_tree(right_data,attribute_list,right_values)
            else:
                root.right=getLeaf(root.attribute,root.value)
            root.left=getLeaf(root.attribute,root.value)
        else:
            if information_gain!=0:
                root.left=decision_tree(left_data,attribute_list,left_values)
            else:
                root.left=getLeaf(root.attribute,root.value)
            root.right=getLeaf(root.attribute,root.value)
        return root

    else:
        return None

######################################
#
#
######################################
def chiSquarePruning(tree,input_list,threshold):
    if tree.left != None and tree.right != None:
        separated_data=split_data(input_list,tree.attribute,tree.threshold)
        chiSquarePruning(tree.left,separated_data[0],threshold)
        chiSquarePruning(tree.right,separated_data[1],threshold)
        class_values=dict()
        left_values=separated_data[2]
        right_values=separated_data[3]
        op=len(input_list[0])-1
        for row in input_list:
            addData(class_values,row[op])

        for key in class_values.keys():
            if key not in left_values:
                left_values[key] = 0
            if key not in right_values:
                right_values[key] = 0

        total_left=sum(left_values.values())
        total_right=sum(right_values.values())
        left_chisquare=0
        right_chisquare=0

        if total_left != 0:
            for key in left_values:
                expected_value= (total_left/(total_left+total_right))*class_values[key]
                left_chisquare += math.pow((left_values[key]-expected_value),2)/expected_value

        if total_right != 0:
            for key in right_values:
                expected_value= (total_right/(total_left+total_right))*class_values[key]
                right_chisquare += math.pow((right_values[key]-expected_value),2)/expected_value

        if (left_chisquare+right_chisquare)<threshold:
            tree.left=None
            tree.right=None

######################################
#
#
######################################

def getPlotList(tree,input_list,level,op):

    if tree.left != None and tree.right != None:
        separated_data=split_data(input_list,tree.attribute,tree.threshold)
        left_data=separated_data[0]
        right_data=separated_data[1]
        x=0
        y=0
        width=0
        height=0
        if tree.attribute==0:
            x=tree.threshold
            y=min(min(left_data,key=lambda x : x[1])[0],min(right_data,key=lambda x : x[1])[0])
            height=max(float(max(left_data,key=lambda x : x[1])[0]),float(max(right_data,key=lambda x : x[1])[0]))-float(y)
            width=math.fabs(float(min(left_data,key=lambda x : x[0])[0])-x)
            op.append([x,y,-width,height,level])
            #print(x,y,-width,height,"first")
            width=math.fabs(float(max(left_data,key=lambda x : x[0])[0]) - float(x))
            #print(x,y,width,height,"first")
            op.append([x,y,width,height,level])
        else:
            y=tree.threshold
            x=min(min(left_data,key=lambda x : x[0])[0],min(right_data,key=lambda x : x[0])[0])
            width=max(float(max(left_data,key=lambda x : x[0])[0]),float(max(right_data,key=lambda x : x[0])[0]))-float(x)
            height=math.fabs(float(min(left_data,key=lambda x : x[1])[0])-y)
            op.append([x,y,width,-height,level])
            #print(x,y,width,-height,"Second")
            height=math.fabs(float(max(left_data,key=lambda x : x[1])[0]) - float(y))
            #print(x,y,width,height,"Second")
            op.append([x,y,width,height,level])

        getPlotList(tree.left,left_data,level+1,op)
        getPlotList(tree.right,right_data,level+1,op)

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

def plotSpace(list):
    plot_list=[]
    color=['r','g','b','violet','gray','cyan','m','brown','orange','purple']
    for item in list:
        plot_list.append(patches.Rectangle((float(item[0]),float(item[1])),float(item[2]),float(item[3]),fill=False,edgecolor=color[int(item[4])]))
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    for p in plot_list:
        ax6.add_patch(p)
    plt.show()

######################################
#  main()
######################################
def main():
    attributes=[]
    filename="train_data.csv"
    significance_level=5
    significance = {1:6.635,2.5:5.024,5:3.841,10:2.706,90:0.016,95:0.004,99:0.00}
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
    chiSquarePruning(tree,list,significance[significance_level])
    op=[]
    getPlotList(tree,list,0,op)
    plotSpace(op)
    #tree.printStatistics("Decision Tree")
    #tree.export_configuration("decisionTree")
    #print(op)

    #tree.printStatistics("Pruned Tree")
    #tree.export_configuration("prunedTree")

if __name__ == "__main__":
    main()