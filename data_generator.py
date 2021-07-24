# Imports ---------------------------------------------------------------------
import pandas as pd
import random
import os
from subprocess import Popen, PIPE, DEVNULL

# Global Variables ------------------------------------------------------------
data_dir = "global_data/"
data = []

# Expanded Dataset ------------------------------------------------------------
def gen_rows(df, count, idx):
    for i in range(count):
    
        # Generate random values
        row = []
        for j in range(6):
            row.append(random.uniform(0, 1))
        
        # Bin values and calculate loss
        
        # Occupancy
        loss = 1
        if row[0] > 0.35:
            row[0] = "SingleFamily"
            loss *= 1/1.2
        else:
            row[0] = "MultiFamily"
            loss *= 1
        
        # Constructioni
        if row[1] > 0.5:
            row[1] = "Frame"
            loss *= 1
        elif row[1] > 0.2:
            row[1] = "Masonry"
            loss *= 0.8
        else:
            row[1] = "Concrete"    
            loss *= 0.7    
        
        # Year
        if row[2] > 0.5:
            row[2] = "<=1995"
            loss *= 1
        elif row[2] > 0.2:
            row[2] = "1995-2005"
            loss *= 0.9
        else:
            row[2] = "2005+"
            loss *= 0.6
        
        # Floor
        if row[3] >= 0.5:
            row[3] = "1"
            loss *= 0.8
        elif row[3] > 0.2:
            row[3] = "2+"
            loss *= 1
        else:
            row[3] = "0"
            loss *= 0.9
        
        # Square Footage
        if row[4] > 0.7:
            row[4] = "1700+"
            loss *= 1
        elif row[4] > 0.3:
            row[4] = "1500-1700"
            loss *= 0.8
        else:
            row[4] = "<1500"
            loss *= 0.6
        
        # Windspeed
        if row[5] > 0.984375:
            row[5] = "7"
            loss *= 1
        elif row[5] > 0.96875:
            row[5] = "6"
            loss *= 0.9
        elif row[5] > 0.9375:
            row[5] = "5"
            loss *= 0.81
        elif row[5] > 0.875:
            row[5] = "4"
            loss *= 0.729
        elif row[5] > 0.75:
            row[5] = "3"
            loss *= 0.6561
        elif row[5] > 0.5:
            row[5] = "2"
            loss *= 0.59049
        else:
            row[5] = "1"
            loss *= 0.531441
        
        # Append data
        df.loc[idx+i] = [idx+i+1] + row + [loss]
        
        if i % 1000 == 0:
            print("Completed", i, "rows.")
    
    return df

def gen_expanded_dataset():
#    cols = ["Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage", "Windspeed", "Loss"]
    
    # Create directory
    if not os.path.isdir("%sdataset_expanded/"%data_dir):
        os.mkdir("%sdataset_expanded/"%data_dir)
    
    # Generate data
    for k in ["train", "val", "test"]:
        
        # Load data
        df = pd.read_csv("%sdataset_base/%s.csv"%(data_dir, k))
        df = df[["Property", "Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage", "Windspeed", "Loss"]]
#        df.reset_index("Property")
        print(df)
        
        # Get counts
        idx = df.shape[0]
        if k == "train":
            rows = 70000 - idx
        else:
            rows = 15000 - idx
        
        # Call function
        df = gen_rows(df, rows, idx)
        print(df)
        
        # Save to file
        df.to_csv("%sdataset_expanded/%s.csv"%(data_dir, k))

# Call function
#gen_expanded_dataset()
        
# Wind Dataset ----------------------------------------------------------------
def gen_wind_dataset():
    pass

df = pd.DataFrame([[23.4, 75.7], 
                   [29.3, 89.6], 
                   [35.6, 88.0], 
                   [38.6, 85.3]], columns=["glat", "glon"])
df.to_csv("test_points.csv")

r_dir = "C:/Program Files/R/R-4.0.3/bin/Rscript"
p = Popen([r_dir,      # Note: Occasionally generates error (OSError: [WinError 6] The handle is invalid) and requires kernel restart. Something about subprocess trying to cleanup previous subprocess executions.
           "RGenWindData.R", 
           "test_points.csv", 
           "Katrina-2005", 
           "test_wind.csv", 
           ""
           ], 
          stdout=PIPE, 
          stderr=PIPE, 
          stdin=DEVNULL
          )
(out, err) = p.communicate();
print ("\ncat returned code = %d\n" % p.returncode)
print ("cat output:\n%s\n" % out)
print ("cat errors:\n%s\n" % err)