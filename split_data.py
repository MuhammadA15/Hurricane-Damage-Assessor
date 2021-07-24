# Imports ---------------------------------------------------------------------
import numpy as np
import pandas as pd

# Load Data -------------------------------------------------------------------
data = None
d = ""
with open("%sdata/VulnerabilityData_csv.csv"%d) as file:
    data = pd.read_csv(file)

# Split Data ------------------------------------------------------------------

# Get indiceis
l = len(data)
l1 = int(l*0.7)
l2 = l1 + int(l*0.15)
print("Indicies: \nTrain:\t%d\t%d\nVal:\t%d\t%d\nTest:\t%d\t%d\n"%(0, l1, l1, l2, l2, l))

# Split dataframe
train = data.iloc[:l1]
val = data.iloc[l1:l2]
test = data.iloc[l2:]

# Save To File ----------------------------------------------------------------
train.to_csv("%strain.csv"%dir)
val.to_csv("%sval.csv"%dir)
test.to_csv("%stest.csv"%dir)