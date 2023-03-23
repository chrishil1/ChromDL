import pandas as pd
import numpy as np
from bstrap import bootstrap
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric', default='auROC')
parser.add_argument('-r', '--runs', type=int, default=100000) 

args = parser.parse_args()

# Define metrid
metric = np.mean

# Load in auROC/auPRC metrics
df = pd.read_csv("auroc_auprcs.csv")

# Specify method and comparison column
if args.metric == "auROC":
    method = ["DeepSEA_auROC", "DanQ_auROC", "DanQ-JASPAR_auROC"]
    comp = "ChromDL_auROC"
elif args.metric == "auPRC":
    method = ["DeepSEA_auPRC", "DanQ_auPRC", "DanQ-JASPAR_auPRC"]
    comp = "ChromDL_auPRC"
else:
    print("Incorrect trial: enter \'auROC\' or \'auPRC\'")
    sys.exit()

# Go through each comparison method
for x in method:
    data_method1 = df[comp]
    data_method2 = df[x]

    # Calculate Confidence intervals, p-values for nbr_trials
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=args.runs)
    print(stats_method1)
    print(stats_method2)
    print(f"p_value: {p_value}")
