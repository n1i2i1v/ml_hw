from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', nargs ='+')
    args = vars(parser.parse_args())
    return args


arg = parse_args()["f"][0]
for x in sys.stdin:
	str_x = str(x[:5])
	x_flt = float(str_x)
	if arg[2] == "+":
		print(int(arg[0])*x_flt+int(arg[3]))
	elif arg[2] == "-":
		print(int(arg[0])*x_flt-int(arg[3]))
	elif arg[2] == "^" and arg[4] == "+":
		print(int(arg[0])*x_flt**int(arg[3])+int(arg[5]))
	elif arg[2] == "^" and arg[4] == "-":
		print(int(arg[0])*x_flt**int(arg[3])-int(arg[5]))
	else:
		print(int(arg[0])*x_flt)


