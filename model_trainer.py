# Imports ---------------------------------------------------------------------

# File handling
import os
import pickle
import shutil

# Data handling
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# ML
import tensorflow as tf

# Class Definition ------------------------------------------------------------
class Trainer():
    
    def __init__(self, dataset, d=""):
        
        # Class variables.
        self.dataset = dataset
        self.dir = d
        self.global_data_dir = "global_data/"
        self.user_data_dir = "%sdata/"%self.dir
        self.model_dir = "%smodel_files/"%self.dir
        self.training_dir = "%straining_files/"%self.model_dir
        self.ckpt_dir = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.encoder = None
        self.model_list = {}
        
        # Get data
        self.load_list()
        self.load_data()
    
    # File Management ---------------------------------------------------------
    def load_list(self):
        if os.path.isfile("%smodel_list.pickle"%self.training_dir):
            with open("%smodel_list.pickle"%self.training_dir, "rb") as f:
                self.model_list = pickle.load(f)
        else:
            self.model_list = []
        
    def load_data(self):
        print("Loading data...")
        self.global_data_dir = "%sdataset_%s/"%(self.global_data_dir, self.dataset)
        
        if self.dataset == "base" or self.dataset == "expanded":
        
            if os.path.isfile("%strain_x.pickle"%self.global_data_dir):
                with open("%strain_x.pickle"%self.global_data_dir, "rb") as f:
                    self.train_x = pickle.load(f)
                with open("%strain_y.pickle"%self.global_data_dir, "rb") as f:
                    self.train_y = pickle.load(f)
                with open("%sval_x.pickle"%self.global_data_dir, "rb") as f:
                    self.val_x = pickle.load(f)
                with open("%sval_y.pickle"%self.global_data_dir, "rb") as f:
                    self.val_y = pickle.load(f)
                with open("%stest_x.pickle"%self.global_data_dir, "rb") as f:
                    self.test_x = pickle.load(f)
                with open("%stest_y.pickle"%self.global_data_dir, "rb") as f:
                    self.test_y = pickle.load(f)
                print("Data loaded")
            
            else:
                train = None
                val = None
                test = None
                with open("%strain.csv"%self.global_data_dir) as file:
                    train = pd.read_csv(file)
                    train = train[["Property", "Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage", "Windspeed", "Loss"]]
                    train.set_index("Property", inplace=True)
                with open("%sval.csv"%self.global_data_dir) as file:
                    val = pd.read_csv(file)
                    val = val[["Property", "Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage", "Windspeed", "Loss"]]
                    val.set_index("Property", inplace=True)
                with open("%stest.csv"%self.global_data_dir) as file:
                    test = pd.read_csv(file)
                    test = test[["Property", "Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage", "Windspeed", "Loss"]]
                    test.set_index("Property", inplace=True)
            
                # Split data
                print("Preprocessing data...")
                self.train_x = train[["Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage"]]
                train_x_wind = train[["Windspeed"]]
                self.train_y = train[["Loss"]]
                self.val_x = val[["Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage"]]
                val_x_wind = val[["Windspeed"]]
                self.val_y = val[["Loss"]]
                self.test_x = test[["Occupancy", "Construction", "YearBuilt", "Floor", "SquareFootage"]]
                test_x_wind = test[["Windspeed"]]
                self.test_y = test[["Loss"]]
                print("Data loaded")
                
                # Encode data
                self.preprocess_insurance_data(train_x_wind, val_x_wind, test_x_wind)
                print("Data preprocessed")
                
        elif self.dataset == "wind":
            self.train = {}
            for file in os.listdir(self.global_data_dir):
                if file[-11:] != "lat_lon.csv":
                    k = file.split("_")[0]
                    df = pd.read_csv("%s%s"%(self.global_data_dir, file))
                    y = df[["vmax_gust"]]
                    x = df.drop(["Unnamed: 0", "X", "record_identifier", "max_sus_wind", "mins", "latitude", "longitude", "gust_dur", "sust_dur"], axis=1)
                    
                    if k == "Ivan-2004":
                        self.val_x = x
                        self.val_y = y
                    elif k == "Florence-2018":
                        self.test_x = x
                        self.test_y = y
                    else:
                        self.train.update({k: {"x": x, "y": y}})
            print(self.test_x.columns)
            
            self.preprocess_wind_data()
                
    
    def write_dataset_to_pickle(self):
        print("Writing data to pickle files...")
        with open("%strain_x.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.train_x, f)
        with open("%strain_y.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.train_y, f)
        with open("%sval_x.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.val_x, f)
        with open("%sval_y.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.val_y, f)
        with open("%stest_x.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.test_x, f)
        with open("%stest_y.pickle"%(self.global_data_dir), "wb") as f:
            pickle.dump(self.test_y, f)
        print("Pickle files written")
    
    def update_data(self, d):
        self.dataset = d
        self.global_data_dir = "global_data/dataset_%s/"%self.dataset
        self.load_data()
    
    # Preprocess Data ---------------------------------------------------------
    def encode_insurance(self, x, wind, y):
        x_hot = self.encoder.transform(x).toarray()
        wind = wind.to_numpy()
    
        x = []
        for i, r in enumerate(x_hot):
            row = []
            for j in r:
                row.append(j)
            row.append(wind[i][0]*1.0)
            x.append(row)
    
        y_n = []
        for r in np.ndarray.tolist(y.to_numpy()):
            y_n.append(r[0])
    
        return x, y_n
        print("Data written to files")
    
    def preprocess_insurance_data(self, train_x_wind, val_x_wind, test_x_wind):
        
        # Encode data
        print("Encoding data...")
        self.encoder = OneHotEncoder()
        self.encoder.fit(self.train_x)
        with open("%sencoder-one_hot.pickle"%self.training_dir, "wb") as f:
            pickle.dump(self.encoder, f)
        self.train_x, self.train_y = self.encode_insurance(self.train_x, train_x_wind, self.train_y)
        self.val_x, self.val_y = self.encode_insurance(self.val_x, val_x_wind, self.val_y)
        self.test_x, self.test_y = self.encode_insurance(self.test_x, test_x_wind, self.test_y)
        print("Data encoded")
        
        # Pickle data
        self.write_dataset_to_pickle()
    
    def encode_wind_column(self, enc, df, col):
        vals = [[x] for x in list(df[col].values)]
        ohe = enc.transform(vals).toarray()
        ohe = pd.DataFrame(ohe, columns=["%s_%d"%(col, i) for i in range(len(ohe[0]))])
        df = pd.concat([df, ohe], axis=1).drop([col], axis=1)
        return df
    
    def preprocess_wind_data(self):
        print("Encoding data...")
        enc_record_ids = OneHotEncoder(handle_unknown="ignore")
        enc_record_ids.fit([["C"], ["I"], ["L"], ["P"], ["R"], ["S"], ["T"]])
        enc_sys_stats = OneHotEncoder(handle_unknown="ignore")
        enc_sys_stats.fit([["TS"], ["HU"], ["EX"], ["SD"], ["SS"], ["LO"], ["WV"], ["DB"]])
        
        for k in self.train:
#            self.train[k] = self.encode_wind_column(enc_record_ids, self.train[k], "record_identifier")
            print(type(self.train[k]["x"]))
            self.train[k]["x"] = self.encode_wind_column(enc_sys_stats, self.train[k]["x"], "sys_status")
        
#        self.val_x = self.encode_wind_column(enc_record_ids, self.val_x, "record_identifier")
        self.val_x = self.encode_wind_column(enc_sys_stats, self.val_x, "sys_status")
#        self.test_x = self.encode_wind_column(enc_record_ids, self.test_x, "record_identifier")
        self.test_x = self.encode_wind_column(enc_sys_stats, self.test_x, "sys_status")
        print("Data encoded")
        
    # Model Training ----------------------------------------------------------
    def wind_data_generator(self, batch_size=None):
        x_batch = []
        y_batch = []
        
        
    def train_model_tf(self, model_key, layers, epochs=100, batch_size=None, drop_remain=True, save=True):
        
        # Build model
        model = tf.keras.Sequential([tf.keras.layers.Input((len(self.train_x[0]), ))])
        for l in layers:
            model.add(l)
        model.summary()
        model.compile(optimizer="adam", 
                      loss=tf.keras.losses.MeanSquaredError(), 
                      metrics=["mean_absolute_error"]
                      )
        
        # Set up checkpoints
        ckpt_dir = "%scheckpoints/%s/"%(self.training_dir, model_key)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = "%scp-{epoch:04d}.hdf5"%ckpt_dir
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, 
                                                         verbose=1, 
                                                         save_weights_only=True,
                                                         save_freq="epoch"
                                                         )
        
        # Make datasets
        if batch_size is not None:
            f = tf.constant(self.train_x)
            l = tf.constant(self.train_y)
            train_dataset = tf.data.Dataset.from_tensor_slices((f, l))
            f = tf.constant([self.val_x])
            l = tf.constant([self.val_x])
            val_dataset = tf.data.Dataset.from_tensor_slices((f, l))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=drop_remain)
        
            # Train
            history = model.fit(train_dataset, 
                                epochs=epochs, 
                                validation_data=val_dataset, 
                                callbacks=[cp_callback]
                                )
        else:
            history = model.fit(self.train_x, 
                                self.train_y, 
                                epochs=epochs, 
                                validation_data=(self.val_x, self.val_y), 
                                callbacks=[cp_callback]
                                )
        
        # Evaluate for best results on validation data
        best = [-1, 1]
        for k in history.history:
            for i, r in enumerate(history.history[k]):
                if k == "val_mean_absolute_error" and r < best[1]:
                    best[0] = i+1
                    best[1] = r
        print("\nPlot:")
        plt.plot([i+1 for i in range(len(history.history["mean_absolute_error"]))], history.history["mean_absolute_error"])
        plt.title("Training for %s"%model_key)
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.show()
        print("Best MAE at Epoch %d: %f\n\n"%(best[0], best[1]))
#        shutil.copy("%scp-%04d.hdf5"%(ckpt_dir, best[0]), "%smodel_3nn.hdf5"%(self.model_dir, best[0]))
        
        # Evaluate model on test data
        model.load_weights("%scp-%04d.hdf5"%(ckpt_dir, best[0]))
        test_results = model.evaluate(self.test_x, self.test_y)
        print("Test Loss, Test MAE:", test_results)
        
        # Update list and save
        if save is True:
            self.model_list.append({"key": model_key, 
                                          "best_epoch": best[0], 
                                          "best_epoch_mae": best[1], 
                                          "test_loss": test_results[0], 
                                          "test_mae": test_results[1], 
                                          "plot": {"x": [i+1 for i in range(len(history.history["mean_absolute_error"]))], "y": history.history["mean_absolute_error"]}, 
                                          "dataset": self.dataset, 
                                          "hyperparameters": {"epochs": epochs, 
                                                              "batch_size": batch_size, 
                                                              "drop_remain": drop_remain
                                                              }
                                          })
            with open("%smodel_list.pickle"%self.training_dir, "wb") as f:
                pickle.dump(self.model_list, f)
        model.save("%smodel_%s.hdf5"%(self.training_dir, model_key), save_format="hdf5")
    
    def update_model_files(self):
        best = {"key": "DEFAULT", 
                "test_mae": 1, 
                }
        for k in self.model_list:
            if self.model_list[k]["test_mae"] < best["test_mae"]:
                best = self.model_list[k]
                best.update({"key": k})
        shutil.copy("%smodel_%s.hdf5"%(self.training_dir, best["key"]), "%smodel.hdf5"%self.model_dir)
        shutil.copy("%sencoder-one_hot.pickle"%self.training_dir, "%sencoder-one_hot.pickle"%self.model_dir)
    
    def prev_experiments(self):
        
        # Get previous models
        self.load_list()
                
        # Display
        best = {"key": "DEFAULT", 
                "test_mae": 1, 
                }
        print("Previously Trained Models:")
        for r in self.model_list:
            if r["test_mae"] < best["test_mae"]:
                best = r
            print("%s: -------------------------------------------------------------------------------"%r["key"])
            for k_r in r:
                if k_r == "plot":
                    plt.plot(r[k_r]["x"], r[k_r]["y"])
                    plt.title("Training for %s"%r["key"])
                    plt.xlabel("Epoch")
                    plt.ylabel("MAE")
                    plt.show()
                elif r == "key":
                    continue
                else:
                    print("%s:"%k_r, r[k_r])
            print()
        print("Best:")
        for k in best:
            if k != "plot":
                print("%s:"%k, best[k])
    
    def run_experiments(self, new_models, save=True):
        
        # Get previous models
        self.load_list()
        
        # Experiment on new models
        for k in new_models:
            files = []
            for f in os.listdir(self.training_dir):
                if k in f:
                    files.append(f)
            if k[0:2] == "tf":
                self.train_model_tf("%s-%d"%(k, len(files)), 
                                    new_models[k]["model"], 
                                    new_models[k]["epochs"], 
                                    new_models[k]["batch_size"],
                                    new_models[k]["drop_remain"], 
                                    save=save)
        
        # Save results
        with open("%smodel_list.pickle"%self.training_dir, "wb") as f:
            pickle.dump(self.model_list, f)

# Test ------------------------------------------------------------------------
new_models = {}
#new_models.update({"tf_1_nn_128": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                             tf.keras.layers.Dense(1)
#                                             ], 
#                                   "epochs": 100, 
#                                   "batch_size": None, 
#                                   "drop_remain": False}})
#new_models.update({"tf_2_nn_128_64": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                tf.keras.layers.Dense(64, activation="relu"), 
#                                                tf.keras.layers.Dense(1)
#                                                ], 
#                                      "epochs": 100, 
#                                      "batch_size": None, 
#                                      "drop_remain": False}})
#new_models.update({"tf_3_nn_128_64_32": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                   tf.keras.layers.Dense(64, activation="relu"), 
#                                                   tf.keras.layers.Dense(32, activation="relu"), 
#                                                   tf.keras.layers.Dense(1)
#                                                   ], 
#                                         "epochs": 100, 
#                                         "batch_size": None, 
#                                         "drop_remain": False}})
#new_models.update({"tf_1_nn_128": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                             tf.keras.layers.Dense(1)
#                                             ], 
#                                   "epochs": 100, 
#                                   "batch_size": 7000, 
#                                   "drop_remain": False}})
#new_models.update({"tf_2_nn_128_64": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                tf.keras.layers.Dense(64, activation="relu"), 
#                                                tf.keras.layers.Dense(1)
#                                                ], 
#                                      "epochs": 100, 
#                                      "batch_size": 7000, 
#                                      "drop_remain": False}})
#new_models.update({"tf_3_nn_128_64_32": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                   tf.keras.layers.Dense(64, activation="relu"), 
#                                                   tf.keras.layers.Dense(32, activation="relu"), 
#                                                   tf.keras.layers.Dense(1)
#                                                   ], 
#                                         "epochs": 100, 
#                                         "batch_size": 7000, 
#                                         "drop_remain": False}})
#new_models.update({"tf_1_nn_128": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                             tf.keras.layers.Dense(1)
#                                             ], 
#                                   "epochs": 1100, 
#                                   "batch_size": 2000, 
#                                   "drop_remain": False}})
#new_models.update({"tf_2_nn_128_64": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                tf.keras.layers.Dense(64, activation="relu"), 
#                                                tf.keras.layers.Dense(1)
#                                                ], 
#                                      "epochs": 100, 
#                                      "batch_size": None, 
#                                      "drop_remain": False}})
#new_models.update({"tf_3_nn_128_64_32": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                                   tf.keras.layers.Dense(64, activation="relu"), 
#                                                   tf.keras.layers.Dense(32, activation="relu"), 
#                                                   tf.keras.layers.Dense(1)
#                                                   ], 
#                                         "epochs": 100, 
#                                         "batch_size": 2000, 
#                                         "drop_remain": False}})
#new_models.update({"tf_test": {"model": [tf.keras.layers.Dense(128, activation="relu"), 
#                                   tf.keras.layers.Dense(64, activation="relu"), 
#                                   tf.keras.layers.Dense(32, activation="relu"), 
#                                   tf.keras.layers.Dense(1)
#                                   ], 
#                         "epochs": 1, 
#                         "batch_size": 3000, 
#                         "drop_remain": True}})
#new_models.update({"tf_3_nn_64_32_15_dropout": {"model": [tf.keras.layers.Dense(64, activation="relu"), 
#                                                         tf.keras.layers.Dropout(0.1, seed=5),
#                                                         tf.keras.layers.Dense(32, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.1, seed=7), 
#                                                         tf.keras.layers.Dense(15, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.1, seed=2),
#                                                         tf.keras.layers.Dense(1)
#                                                         ],
#                                                "epochs": 100, 
#                                                "batch_size": None,
#                                                "drop_remain": False}})
#new_models.update({"tf_3_nn_64_32_15_dropout_0.2": {"model": [tf.keras.layers.Dense(64, activation="relu"), 
#                                                         tf.keras.layers.Dropout(0.2, seed=5),
#                                                         tf.keras.layers.Dense(32, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.2, seed=7), 
#                                                         tf.keras.layers.Dense(15, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.2, seed=2),
#                                                         tf.keras.layers.Dense(1)
#                                                         ],
#                                                "epochs": 100, 
#                                                "batch_size": None,
#                                                "drop_remain": False}})
#new_models.update({"tf_3_nn_64_32_15_dropout_0.2": {"model": [tf.keras.layers.Dense(64, activation="relu"), 
#                                                         tf.keras.layers.Dropout(0.05, seed=5),
#                                                         tf.keras.layers.Dense(32, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.05, seed=7), 
#                                                         tf.keras.layers.Dense(15, activation="relu"),
#                                                         tf.keras.layers.Dropout(0.05, seed=2),
#                                                         tf.keras.layers.Dense(1)
#                                                         ],
#                                                "epochs": 100, 
#                                                "batch_size": None,
#                                                "drop_remain": False}})
new_models.update({"tf_3_nn_64_32_": {"model": [tf.keras.layers.Dense(64, activation="relu"), 
                                                   tf.keras.layers.Dense(32, activation="relu"), 
                                                   tf.keras.layers.Dense(15, activation="relu"), 
                                                   tf.keras.layers.Dense(1)
                                                   ], 
                                         "epochs": 100, 
                                         "batch_size": None, 
                                         "drop_remain": False}})
print("Building...")
mt = Trainer("wind", "users/MASTER_USER/")
print("Training...")
##test = [tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(1)]
##mt.train_model_tf("test", test, epochs=20, batch_size=2000, drop_remain=False)
#mt.run_experiments(new_models, save=False)