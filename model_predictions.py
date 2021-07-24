# Imports ---------------------------------------------------------------------

# Files
import os
import pickle
import shutil

# Data
import pandas as pd

# Geocoding
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
import time

# Machine learning
from model_trainer import Trainer
from subprocess import Popen, PIPE, DEVNULL
import tensorflow as tf

# Class Definition ------------------------------------------------------------
class HurricaneLosses():

    def __init__(self, u, d=""):
        
        # Class variables
        self.user = u
        self.dir = d
        self.user_dir = "%susers/%s/"%(self.dir, self.user)
        self.data_dir = "%sdata/"%self.user_dir
        self.model_dir = "%smodel_files/"%self.user_dir
#        self.r_dir = "C:/Users/sali/Documents/R/R-4.0.3/bin/Rscript"
        self.r_dir = "C:/Program Files/R/R-4.0.3/bin/Rscript"
        self.locator = None
        self.encoder = None
        self.model = None
        self.policies = pd.DataFrame(columns=["policynumber", 
                                              "streetname", 
                                              "city", 
                                              "statecode", 
                                              "postalcode", 
                                              "cntrycode", 
                                              "occtype", 
                                              "bldgclass", 
                                              "numfloors", 
                                              "yearbuilt", 
                                              "floorarea", 
                                              "address", 
                                              "codelevel", 
                                              "glat", 
                                              "glon", 
                                              "county"
                                              ])
        self.policies.set_index("policynumber", inplace=True)
        self.located = None
        
        # Build starting folders and files
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists("%straining_files/"%self.model_dir):
            os.makedirs("%straining_files/"%self.model_dir)
        shutil.copy("%susers/MASTER_USER/model_files/encoder-one_hot.pickle"%self.dir, "%sencoder-one_hot.pickle"%self.model_dir)
        shutil.copy("%susers/MASTER_USER/model_files/model.hdf5"%self.dir, "%smodel.hdf5"%self.model_dir)
        
        # Class function calls
#        self.train_model()
        self.load_models()
        
    # Admin Functions ---------------------------------------------------------
    def set_directory(self, d):
        
        # Change folder namesif not os.path.exists(ckpt_dir):
        os.rename(self.user_dir, "%susers/%s/"%(d, self.user))
        
        # Change variable names
        self.dir = d
        self.user_dir = "%susers/%s/"%(self.dir, self.user)
        self.data_dir = "%sdata/"%self.user_dir
        self.model_dir = "%smodel_files/"%self.user_dir
        
    def renname_user(self, u):
        self.user = u
        self.set_directory(self.dir)
    
    def set_R_directory(self, d):
        self.r_dir = d
    
    def save_class(self, directory, file_name="HurricaneLoss"):
        with open("%s%s.pickle"%(directory, file_name), "wb") as f:
            pickle.dump(self, f)
    
    def load_models(self):
        print("Loading models...")
        
        # Load geocoder
        self.locator = Nominatim(user_agent="myGeocoder")
        print("Locator loaded")

        # Load one-hot-encoder
        self.encoder = None
        with open("%sencoder-one_hot.pickle"%self.model_dir, "rb") as f:
            self.encoder = pickle.load(f)
        print("Encoder loaded")
        
        # Load model
        self.model = tf.keras.models.load_model("%smodel.hdf5"%self.model_dir)
        print("Model loaded")
        
        print("All models loaded")
    
    # Train New Model ---------------------------------------------------------
    def train_model(self):
        
        # Build model
        model = [tf.keras.layers.Dense(128, activation='relu'),
                 tf.keras.layers.Dense(64, activation='relu'),
                 tf.keras.layers.Dense(32, activation='relu'),
                 tf.keras.layers.Dense(1)
                 ]
    
        # Train model
        trainer = Trainer()
        trainer.train_model_tf("DEFAULT", model, epochs=100, batch_size=7000, drop_remain=False)
        # LOAD SPECIFIC MODEL FILE
        
        # Load new models
        self.load_models()
    
    # Geocoding ---------------------------------------------------------------
    def gcode (self, address, postalcode, lat):
        if (lat=='Yes'):
            if self.locator.geocode(address) is None:
                return self.locator.geocode(postalcode).point[0]
            else:
                return self.locator.geocode(address).point[0]
        else:
            if self.locator.geocode(address) is None:
                return self.locator.geocode(postalcode).point[1]
            else:
                return self.locator.geocode(address).point[1]
            

    def do_geocode(self, address, postalcode):
        try:
            if self.locator.geocode(address) is None:
                return ('Zip', self.locator.geocode(postalcode).latitude, self.locator.geocode(postalcode).longitude)
            else:
                return ('Street', self.locator.geocode(address).latitude, self.locator.geocode(address).longitude)
        
        except GeocoderTimedOut:
            return self.do_geocode(address, postalcode)
        except GeocoderQuotaExceeded:
            time.sleep(15)
            return self.do_geocode(address, postalcode)

    def return_gran(self, address, postalcode, point):
        z=self.do_geocode (address, postalcode)
        if (point==0):
            return (z[0])
        if (point==1):
            return(z[1])
        if (point==2):
            return (z[2])
        if (point==3):
            return ((self.locator.reverse(str(z[1]) +" , " + str(z[2])).raw['address']['county']).replace(' County', ''))

    def geolocate_data(self):

        # Select which cols to use
        add_fields = ['street', 'streetname', 'city', 'state', 'statecode', 'zip', 'zipcode' , 'postalcode' , 'cntrycode']
        found = []
        not_found = []
        for f in add_fields:
            if f in self.df.columns:
                found.append(f)
            else:
                not_found.append(f)
        
        # Check and Clean the Fields that were Found  
        null_fields=[]
        for f in found:
            if (self.df[f].isnull().sum(axis=0)) > 0:
                print (f, self.df[f].isnull().sum(axis=0))
                null_fields.append ([f, self.df[f].isnull().sum(axis=0)])
        for f in null_fields:
            self.df[f[0]].fillna('No ' + f[0], inplace = True)
        
        # Remove all Special Characters
        self.df['streetname']=self.df['streetname'].str.translate({ord(c): None for c in '?!#@#$,.;-@!%^&*)('})
        self.df['city']=self.df['city'].str.translate({ord(c): None for c in '?!@#$,.;-@!#%^&*)('})
        self.df['statecode']=self.df['statecode'].str.translate({ord(c): None for c in '?#!@#$,.;-@!%^&*)('})

        #pad postal codes with zero to the left where only 4 digits 
        self.df['postalcode']=self.df['postalcode'].astype(str).str.pad(5, side='left', fillchar='0')

        # if no Nulls then combine the address field 
        self.df['address'] = self.df['streetname'] + ', ' + self.df['city'] + ', ' + self.df['statecode'] + ', '+ self.df['cntrycode'] + ', ' + self.df['postalcode'].astype(str)

        # check there are no null addresses
        self.df['address']=self.df['address'].fillna(self.df['postalcode'].astype(str) + ', ' + self.df['cntrycode'])

        # Add geolocation
        self.df['codelevel']= self.df.apply(lambda row: self.return_gran (row['address'], row['postalcode'], 0), axis= 1)
        self.df['glat']= self.df.apply(lambda row: self.return_gran (row['address'], row['postalcode'], 1), axis= 1)
        self.df['glon']= self.df.apply(lambda row: self.return_gran (row['address'], row['postalcode'], 2), axis= 1)
        self.df['county']= self.df.apply(lambda row: self.return_gran (row['address'], row['postalcode'], 3), axis= 1)
        
        # Add rowid
        self.df.insert(loc=0, column="rowid", value=[i+1 for i in range(len(self.df.index))])
        self.df.set_index("rowid", inplace=True)

    # Bin Columns -------------------------------------------------------------
    def bin_occupancy(self, x):
        if x < 2:
            return "SingleFamily"
        else:
            return "MultiFamily"

    def bin_construction(self, x):
        if x == 1:
            return "Frame"
        elif x == 2:
            return "Masonry"
        else:
            return "Concrete"

    def bin_numfloors(self, x):
        if x == 0:
            return "0"
        elif x == 1:
            return "1"
        else:
            return "2+"

    def bin_yearbuilt(self, x):
        if x <= 1995:
            return "<=1995"
        elif x >= 2005:
            return "2005+"
        else:
            return "1995-2005"

    def bin_floorarea(self, x):
        if x < 1500:
            return "<1500"
        elif x >= 1700:
            return "1700+"
        else:
            return "1500-1700"
    
    def bin_windspeed(self, x):
        mph = x * 2.23694
        if mph < 50:
            return 0
        elif mph <= 60:
            return 1
        elif mph <= 70:
            return 2
        elif mph <= 90:
            return 3
        elif mph <= 100:
            return 4
        elif mph <= 120:
            return 5
        elif mph <= 140:
            return 6
        else:
            return 7
    
    # Master Policy Management ------------------------------------------------
    def split_prelocated(self):
        self.located = self.policies[self.policies.index.isin(self.df["policynumber"].tolist())]
        self.df = self.df[~self.df["policynumber"].isin(self.located.index.tolist())]

    def update_master_policies(self):
            
        # Add new policies to master df
        self.policies = pd.concat([self.policies, self.df.set_index("policynumber", inplace=False)], ignore_index=False)
                    
        # Attach pre-geolocated policies to newly-geolocated policies
        if self.located is not None:
            self.located.reset_index(inplace=True)
            self.df = pd.concat([self.df, self.located], ignore_index=True)
            self.df.index.name = "rowid"
            self.located = None
        
    # Predict Losses ----------------------------------------------------------
    def preprocess_data(self):
        
        # Bin occtype, bldgclass, numfloors, yearbuilt, and floorarea
        print("Binning columns")
        self.df["occtype"] = self.df.apply(lambda row: self.bin_occupancy(row["occtype"]), axis=1)
        self.df["bldgclass"] = self.df.apply(lambda row: self.bin_construction(row["bldgclass"]), axis=1)
        self.df["numfloors"] = self.df.apply(lambda row: self.bin_numfloors(row["numfloors"]), axis=1)
        self.df["yearbuilt"] = self.df.apply(lambda row: self.bin_yearbuilt(row["yearbuilt"]), axis=1)
        self.df["floorarea"] = self.df.apply(lambda row: self.bin_floorarea(row["floorarea"]), axis=1)
        
        # Geolocate data
        print("Geolocating addresses")
        self.split_prelocated()
        if not self.df.empty:
            self.geolocate_data()
        self.update_master_policies()
        self.df.to_csv("%sGeoAdd_Coded.csv"%self.data_dir)

        # Get windspeeds
        print("Predicting windspeeds")
        p = Popen([self.r_dir,      # Note: Occasionally generates error (OSError: [WinError 6] The handle is invalid) and requires kernel restart. Something about subprocess trying to cleanup previous subprocess executions.
                   "RWindModel.R", 
                   "%sGeoAdd_Coded.csv"%self.data_dir, 
                   "katrina_tracks", 
                   "%sGeoAdd_Wind.csv"%self.data_dir, 
                   ""
                   ], 
                  stdout=PIPE, 
                  stderr=PIPE, 
                  stdin=DEVNULL
                  )
#        (out, err) = p.communicate();
#        print ("\ncat returned code = %d\n" % p.returncode)
#        print ("cat output:\n%s\n" % out)
#        print ("cat errors:\n%s\n" % err)
        while True:
            if os.path.isfile("%sGeoAdd_Wind.csv"%self.data_dir):
                time.sleep(0.1)
                self.df = pd.read_csv("%sGeoAdd_Wind.csv"%self.data_dir)
                break
        self.df["windspeed"] = self.df.apply(lambda row: self.bin_windspeed(row["vmax_gust"]), axis=1)

        # Split dataframe
        x = self.df[["occtype", "bldgclass", "yearbuilt", "numfloors", "floorarea"]]
        x_wind = self.df[["windspeed"]] 
        
        # Encode data
        print("Encoding data")
        x_hot = self.encoder.transform(x).toarray()
        x_wind = x_wind.to_numpy()

        # Attach windspeed
        self.to_pred = []
        for i, r in enumerate(x_hot):
            row = []
            for j in r:
                row.append(j)
            row.append(x_wind[i][0]*1.0)
            self.to_pred.append(row)
        
    def predict_losses(self, input_data_dir):

        # Load data
        self.df = pd.read_csv(input_data_dir)
        self.df.columns = map(str.lower, self.df.columns)
        print("Data loaded")
        
        # Geolocate, get windspeed, and encode
        print("Transforming data")
        self.preprocess_data()

        # Predict loss
        print("Making predictions")
        predictions = [i[0] for i in self.model.predict(self.to_pred)]
        
        # Attach predictions to policies.
        self.df["lossprediction"] = pd.Series(predictions)
        self.df.set_index("policynumber", inplace=True)
        final = self.df[["streetname", 
                           "postalcode", 
                           "city", 
                           "county", 
                           "statecode", 
                           "cntrycode", 
                           "address", 
                           "occtype", 
                           "bldgclass", 
                           "numfloors", 
                           "yearbuilt", 
                           "floorarea", 
                           "codelevel", 
                           "glat", 
                           "glon", 
                           "vmax_gust", 
                           "vmax_sust", 
                           "sust_dur", 
                           "windspeed", 
                           "lossprediction"
                           ]]
        final.to_csv("%sGeoAdd_Predictions.csv"%self.data_dir)
        
        # Cleanup
        del self.df
        del self.to_pred
        os.remove("%sGeoAdd_Coded.csv"%self.data_dir)
        os.remove("%sGeoAdd_Wind.csv"%self.data_dir)
        
        return final
    
# Test ------------------------------------------------------------------------
hl = HurricaneLosses("test_user")
p = hl.predict_losses("users/MASTER_USER/data/GeoAdd.csv")
print(p)
