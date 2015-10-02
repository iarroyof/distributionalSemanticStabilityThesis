from gridObj import *
import argparse

# Yields a generator containing multiple paths for being processed en parallel
def gridGen(fileN, trials):
    grid = gridObj(file = fileN)
    for p in grid.generateRandomGridPaths(trials = trials): # the dict object is put to the output stream.
       yield p
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script generates multiple grid search paths.')
    parser.add_argument('-f', type=str, dest = 'grid_search_file', help='Specifies the grid search complete dictionary.', metavar='FILE')
    parser.add_argument('-t', type=int, dest = 'number_of_trials', metavar = 'N', help = 'Specifies the number of search paths wanted.')
    args = parser.parse_args()

# Create the random grid:
    for path in gridGen(fileN = args.grid_search_file, trials = args.number_of_trials): # the dict object is put to the output stream.
        print path # The paths are append to the stdout.
