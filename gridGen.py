from gridObj import *
import sys


def gridGen(fileN, trials):
    grid = gridObj(file = fileN)
    for p in grid.generateRandomGridPaths(trials = trials): # the dict object is put to the output stream.
       yield p
                
if __name__ == "__main__":
    args = sys.argv
    assert len(args) is 3 # The amount of input args is 3 (the script plus its arguments)?
# Create the random grid:
    grid = gridObj(file = args[1])
    #print grid.generateRandomGridPaths(trials = int(args[2]))
    for p in grid.generateRandomGridPaths(trials = int(args[2])): # the dict object is put to the output stream.
        print p
