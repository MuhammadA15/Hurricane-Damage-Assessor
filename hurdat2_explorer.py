# Imports ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
import os
from global_land_mask import globe
from subprocess import Popen, PIPE, DEVNULL

# Global Variables ------------------------------------------------------------
r_dir = "C:/Program Files/R/R-4.0.3/bin/Rscript"
data_dir = "global_data/dataset_wind/"

# Functions -------------------------------------------------------------------
def get_year(x):
    return int(x[:4])

def get_month(x):
    return int(x[4:6])

def get_day(x):
    return int(x[6:])

def get_hours(x):
    return int(x[:2])

def get_mins(x):
    return int(x[2:])

def get_lat(x):
    val = float(x[:-1])
    d = x[-1]
    if d == "S":
        val *= -1
    return val

def get_lon(x):
    val = float(x[:-1])
    d = x[-1]
    if d == "W":
        val *= -1
    return val

def get_lat_lon_value(x):
    return float(x[:-1])

def get_lat_lon_dir(x):
    return x[-1].upper()

def to_int(x):
    return int(x)

def encode_column(enc, df, col):
    vals = [[x] for x in list(df[col].values)]
    ohe = enc.transform(vals).toarray()
    ohe = pd.DataFrame(ohe, columns=["%s_%d"%(col, i) for i in range(len(ohe[0]))])
    df = pd.concat([df, ohe], axis=1).drop([col], axis=1)
    return df
    # self.df['county']= self.df.apply(lambda row: self.return_gran (row['address'], row['postalcode'], 3), axis= 1)

# Load Data -------------------------------------------------------------------
hurdat = None
with open("global_data/HURDAT2.txt", "r") as f:
    hurdat = f.readlines()
storm_ids = list(pd.read_csv("global_data/storm_ids.csv")["storm_id"].values)
#hurr_tracks = pd.read_csv("global_data/hurr_tracks.csv")

# Convert to Dictionary -------------------------------------------------------
data = {}
i = 0
k = None
while True:
    row = hurdat[i].split(",")
    row = [x.strip() for x in row]
    row = row[:-1]
    
    if len(row) == 3:
        name = row[1]
        basin = row[0][:2]
        num = row[0][2:4]
        year = row[0][4:8]
        k = name + "_" + basin + "_" + year + "_" + num
        data.update({k: []})
    else:
        data[k].append([x.strip() for x in row])
    
    i += 1
    if len(hurdat) == i:
        break
    
#tracks = {}
#for k in storm_ids:
#    tracks.update({k: hurr_tracks.loc[hurr_tracks["storm_id"] == k]})
#    tracks[k].drop("storm_id", axis=1, inplace=True)
#del hurr_tracks

# Get One-Hot Encoders --------------------------------------------------------
enc_record_ids = OneHotEncoder(handle_unknown="ignore")
enc_record_ids.fit([["C"], ["I"], ["L"], ["P"], ["R"], ["S"], ["T"]])
enc_sys_stats = OneHotEncoder(handle_unknown="ignore")
enc_sys_stats.fit([["TS"], ["HU"], ["EX"], ["SD"], ["SS"], ["LO"], ["WV"], ["DB"]])

# Convert Data to DataFrame ---------------------------------------------------
lens = set()
hurdat = {}
for idx, k in enumerate(data):
#    if k[:-11] == "UNNAMED":
#        continue
    name = k.split("_")[0].lower().capitalize() + "-" + k.split("_")[2]
    if name not in storm_ids:
        continue
#    if k[:-11] != "KATRINA":
#        continue
#    if k[-7:-3] != "2005":
#        continue
#    print(k)
    hurdat[k] = pd.DataFrame(data[k], columns=["date", "hours_mins", "record_identifier", "sys_status", "latitude", "longitude", "max_sus_wind", "min_pressure", 
                                                     "wind_34_r_neq", "wind_34_r_seq", "wind_34_r_swq", "wind_34_r_nwq", 
                                                     "wind_50_r_neq", "wind_50_r_seq", "wind_50_r_swq", "wind_50_r_nwq", 
                                                     "wind_64_r_neq", "wind_64_r_seq", "wind_64_r_swq", "wind_64_r_nwq", 
                                                     ])
    
    hurdat[k]["year"] = hurdat[k]["date"].apply(get_year)
    hurdat[k]["month"] = hurdat[k]["date"].apply(get_month)
    hurdat[k]["day"] = hurdat[k]["date"].apply(get_day)
    hurdat[k].drop("date", axis=1, inplace=True)
    
    hurdat[k]["hours"] = hurdat[k]["hours_mins"].apply(get_hours)
    hurdat[k]["mins"] = hurdat[k]["hours_mins"].apply(get_mins)
    hurdat[k].drop("hours_mins", axis=1, inplace=True)
    
#    hurdat[k]["lat_value"] = hurdat[k]["latitude"].apply(get_lat_lon_value)
#    hurdat[k]["lat_dir"] = hurdat[k]["latitude"].apply(get_lat_lon_dir)
#    hurdat[k]["lon_value"] = hurdat[k]["longitude"].apply(get_lat_lon_value)
#    hurdat[k]["lon_dir"] = hurdat[k]["longitude"].apply(get_lat_lon_dir)
#    hurdat[k].drop(["latitude", "longitude"], axis=1, inplace=True)
    hurdat[k]["latitude"] = hurdat[k]["latitude"].apply(get_lat)
    hurdat[k]["longitude"] = hurdat[k]["longitude"].apply(get_lon)
    
    for c in ["max_sus_wind", "min_pressure", "wind_34_r_neq", "wind_34_r_seq", "wind_34_r_swq", "wind_34_r_nwq", "wind_50_r_neq", "wind_50_r_seq", "wind_50_r_swq", "wind_50_r_nwq", "wind_64_r_neq", "wind_64_r_seq", "wind_64_r_swq", "wind_64_r_nwq"]:
        hurdat[k][c] = hurdat[k][c].apply(to_int)
    
    hurdat[k].replace(-999, np.NaN, inplace=True)
    
#    data[k] = encode_column(enc_record_ids, data[k], "record_identifier")
#    data[k] = encode_column(enc_sys_stats, data[k], "sys_status")
    
    locs = []
    for i, row in hurdat[k].iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        neq = row["wind_34_r_neq"]
        seq = row["wind_34_r_seq"]
        nwq = row["wind_34_r_nwq"]
        swq = row["wind_34_r_swq"]
        
        best = 0
        for val in [neq, seq, nwq, swq]:
            if val is not np.NaN:
                if val > best:
                    best = val
        
        side = ((2.0*best*np.sin(45.0))/2.0)/69.0
        for i in range(10):
            a = np.random.uniform(lat-side, lat+side)
            o = np.random.uniform(lon-side, lon+side)
            locs.append(list(row.values) + [a, o])
#        if side > 0:
#            for i in range(10):
#                try:
#                    a = np.random.uniform(lat-side, lat+side)
#                    o = np.random.uniform(lon-side, lon+side)
#                    if globe.is_land(a, o):
#                        locs.append(list(row.values) + [a, o])
#                except:
#                    pass
                
    locs = pd.DataFrame(locs, columns=list(hurdat[k].columns)+["glat", "glon"])
    locs.to_csv(data_dir+name+"_gen_lat_lon.csv")
    
    p = Popen([r_dir,      # Note: Occasionally generates error (OSError: [WinError 6] The handle is invalid) and requires kernel restart. Something about subprocess trying to cleanup previous subprocess executions.
               "RGenWindData.R", 
               "%s%s_gen_lat_lon.csv"%(data_dir, name), 
               name, 
               "%s%s_gen_wind.csv"%(data_dir, name), 
               ""
               ], 
              stdout=PIPE, 
              stderr=PIPE, 
              stdin=DEVNULL
              )
    (out, err) = p.communicate();
#    print ("\ncat returned code = %d\n" % p.returncode)
#    print ("cat output:\n%s\n" % out)
#    print ("cat errors:\n%s\n" % err)
    print(k, "complete")
    

# Display ---------------------------------------------------------------------
