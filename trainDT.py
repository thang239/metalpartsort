####################################################################################
# trainDT.py takes a training data file and produces:
# (a) Two files representing the trained decision tree 1) after automatic induction,
# and 2) after subsequent pruning using the Chi-Squared test. You will later run
# these trees using executeDT.py, described below.2
# (b) Two images illustrating splits in the two decision trees. For each tree, use
# matplotlib to create a plot visualizing how the feature space is partitioned by
# the decision tree, by drawing a box around both regions created at every internal
# (‘split’) node in the tree. You should also show the training samples in the plots.
# (c) On the terminal, print a summary of the contents for each decision tree:
#    i. Number of nodes, and number of leaf (decision) nodes
#    ii. Max, minimum and average depth of root-to-leaf paths in the decision tree
####################################################################################