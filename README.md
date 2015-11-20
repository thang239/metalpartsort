# FILProject
Foundation of Intelligent System Project
Metal Part Sort

1. MLP
1.1 Training
Running command: python3 sdmaze.py puzzle_name with:
    - sdmaze.py: name of main file
    - puzzle_name: path to puzzle file, normally in same directory: 'puzzle1.txt',...'puzzle5.txt'

    Example: python3 sdmaze.py puzzle2.txt

Output: 
 - Each puzzle is solved by A star search with three heuristics function: euclidean, manhattan, manhattan consider orientation
 - The die will be put in S position with 1 on top
 - Each step is a state of maze with the latest number on maze is current position of die with that number is on the top
 - Number of move take from S position to G position and number of node generated/visited will be printed out

Below are command and results from 2 puzzle, puzzle 1 and puzzle 2