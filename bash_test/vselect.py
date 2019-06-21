from argparse import ArgumentParser
import pandas as pd
import numpy as np

def parse_args():
    parser = ArgumentParser(description='Select the first one')
    parser.add_argument('-c', type = int,  help='chooses a column')
    parser.add_argument('d', type=str, help='give a csv')
    args = parser.parse_args()
    return args.c, args.d


def select(data, arg):
    dat = np.array(data)
    n = data.shape[0]
    for i in range(n):
        print(dat[i][arg])

arg, dat = parse_args()
dat_n = pd.read_csv(dat)

select(dat_n, arg)
