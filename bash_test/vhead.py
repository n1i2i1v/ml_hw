from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys

def parse_args():
    parser = ArgumentParser(description='Selects the first n rows')
    parser.add_argument('-n', type = int,  help='number of rows')
    args = parser.parse_args()
    return args.n


arg = parse_args()
i = 0
for x in sys.stdin:
    print(float(x))
    i += 1
    if i>= arg:
	break
	    
