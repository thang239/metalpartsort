# FILProject
Foundation of Intelligent System Project
Metal Part Sort

1. MLP
1.1 Training
Running command: python3 trainLMP.py train_data.csv num_epochs
with:
    - trainMLP.py : name of main file
    - train_data.csv: training data
    - num_epochs: this parameter can be left blank, program will run with default epochs = 10000
    Example: python3 trainLMP.py train_data.csv 1000
Parameters of neural network are hardcoded in source file since this is a simple network:

NUM_LAYERS = 3
NUM_INPUT_NODE = 2
NUM_HIDDEN_NODE = 5
NUM_OUTPUT_NODE = 4
NUM_EPOCHS = 10000
LEARNING_RATE = 0.1

Output: 
    - The file containing neural network weights will be given at specific epochs(0,10,100,1000,10000) in format:
       numEpochs_config
    - After training network, error file and plot of learning curve will be given by name 'error.csv' and 'learning_curve.png'

1.2 Executing
    
