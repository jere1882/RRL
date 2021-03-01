""" Code used to preprocess data released in Carpyncho 

1) persist_all_tiles() must be invoked (only) the very first time you want to use these methods.
   from then on, you may use retrieve_tile(tile_name) to retrieve whatever tile you want to use.
   
2) in order to be able to retrieve subtiles, generate_balanced_subtiles must be invoked (only) the
   very first time you want to use subtiles. From then on, add an optional parameter to retrieve_tile
   to indicate the desired balance.  E.g.:  retrive_tile("b261","1:100")


PATHS:

./tiles/features{i}.pkl     # tiles preprocesadas para cada tile de carpyncho, e.g. featuresb360.pkl
./tiles/label{i}.pkl
./tiles/undersampled_indexes{i}_{rate}  # Para cada tile, subsets de indices. rate es 1/10/100/500/1000. Ejemplo undersampled_indexesb360_10

Análogo para vtiles:

./tiles/features{i}.pkl     # tiles virtuales para i=1...20
./tiles/label{i}.pkl
./tiles/undersampled_indexes{i}_{rate}  # . Ejemplo undersampled_indexes1_10


"""

import carpyncho
client = carpyncho.Carpyncho()

import pandas as pd
from PyAstronomy.pyasl import foldAt
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn import metrics
import pickle
from sklearn.model_selection import StratifiedKFold
import random

invalid = ["b356","others"]
tiles  = [ x for x in list(client.list_tiles()) if not(x in invalid)] # All existing tiles    
    

""" Filter inconvenient sources from a raw tile downloaded from carpyncho, and construct RRL labels """
def preprocess_tile(tile):
    tile = tile[(tile.Mean >= 12) & (tile.Mean <= 16.5)]
    tile = tile.drop(columns=['Meanvariance',
                              'StetsonK',
                              'Freq1_harmonics_rel_phase_0',
                              'Freq2_harmonics_rel_phase_0',
                              'Freq3_harmonics_rel_phase_0'])

    tile = tile.replace([np.inf, -np.inf], np.nan)
    tile = tile.dropna()

    tile = tile[ tile['vs_type'].isin(['RRLyr-RRab','RRLyr-RRc', 'RRLyr-RRd', '']) ]

    features = tile.iloc[:,6:]
    labels = tile['vs_type']   
    labels = pd.get_dummies(labels)
    labels = 1-labels['']
    return features, labels
    
""" Download a tile from carpyncho, filter sources and persist it """    
def persist_tile(tile):
    df = client.get_catalog(tile, "features")
    train_features , train_labels = preprocess_tile(df)
    train_features.to_pickle("tiles/features"+tile+".pkl")  
    train_labels.to_pickle("tiles/label"+tile+".pkl")

""" Download, preprocess and persist all tiles """
def persist_all_tiles():
       
    for tile in tiles:
        try:
            persist_tile(tile)
        except:
            print("Tile "+tile+" failed to be persisted")
            
""" Suffix used to persist balanced subsets of tiles """
def suffix(s):
    if (s=="1:1" or s=="balanced" or s==1):
        return "_1"
    if (s=="1:10" or s==10):
        return "_10"
    if (s=="1:100" or s==100):
        return "_100"
    if (s=="1:500" or s==500):
        return "_500"
    if (s=="1:1000"or s==1000):
        return "_1000"
    if (s=="1:2000" or s=="full" or s==2000):
        return ""
    else:
        return ("_"+str(s))
        
""" Retrieve a previously persisted tile """
def retrieve_tile(tile,rate="full"):
    
    if type(tile) == int or tile.isdigit():  ## RETRIEVE A VIRTUAL TILE
        tile = str(tile)
        features = pd.read_pickle("vtiles/features"+ str(tile) + ".pkl")
        labels   = pd.read_pickle("vtiles/label"+ str(tile) + ".pkl")
        if (rate!="full"):
            with open("vtiles/undersampled_indexes"+tile+suffix(rate)+".pkl", 'rb') as i_file:
                indexes = pickle.load(i_file)
            features = features.loc[indexes,]
            labels   = labels[indexes]
    else:  ## RETRIEVE A REAL TILE
        features = pd.read_pickle("tiles/features"+ tile + ".pkl")
        labels   = pd.read_pickle("tiles/label"+ tile + ".pkl")
    
        if (rate!="full"):
            with open("tiles/undersampled_indexes"+tile+suffix(rate)+".pkl", 'rb') as i_file:
                indexes = pickle.load(i_file)
            features = features.loc[indexes,]
            labels   = labels[indexes]
    return features, labels

""" Generates virtual tiles using a % for each tile """
def generate_virtual_tiles(nsplits=20):

    tiles = list(client.list_tiles())
    
    for tile in tiles:
        if (tile=="others"):
            continue
        
        X,y   = retrieve_tile(tile)
        skf = StratifiedKFold(n_splits=nsplits)
        skf.get_n_splits(X, y)
        
        i=0 
        for train_index, test_index in skf.split(X, y):
            X_test = X.iloc[test_index,:]
            y_test = y.iloc[test_index]
            
            try:
                features = pd.read_pickle("vtiles/features"+str(i) + ".pkl")
                labels   = pd.read_pickle("vtiles/label"+ str(i) + ".pkl")
                features = features.append(X_test)
                labels   = labels.append(y_test)
                features.reset_index(drop=True,inplace=True)
                labels.reset_index(drop=True,inplace=True)
                features.to_pickle("vtiles/features"+str(i)+".pkl") 
                labels.to_pickle("vtiles/label"+str(i)+".pkl") 
            except:
                print("Unable to retrieve vtile "+str(i))
                X_test.to_pickle("vtiles/features"+str(i)+".pkl")
                y_test.to_pickle("vtiles/label"+str(i)+".pkl") 
            
            i=i+1

def get_supported_rates(tile):
    X,y = retrieve_tile(tile)
    max_rate = len(y)/sum(y)
    return([x for x in [1,10,100,150,200,250,350,425,500,750,1000,1250,1500,1750,2000,2250] if x<max_rate])

""" Generated indexes for subtiles by undersampling the negative class """
def generate_balanced_subtiles(virtual=False,nsplits=20):
    
    if virtual:
        ts = [str(t) for t in range(0,nsplits)]
        path = "vtiles/undersampled_indexes"
    else:
        ts = tiles
        path = "tiles/undersampled_indexes"

    for tile in ts:
        X,y   = retrieve_tile(tile)
        rrl_indexes = y[y==1].index.values # Indexes corresponding to RRLS
        rrl_n = len(rrl_indexes)           # Number of RRLS in this tile
        
        norrl_indexes = y[y==0].index.values
        random.shuffle(norrl_indexes)
        
        for rate in get_supported_rates(tile):
            unk_n       = rrl_n * rate   # Expected number of unknown sources in the undersampled DS with rate 'rate'
            rate_ds = rrl_indexes.tolist() + norrl_indexes[0:unk_n].tolist()
            
            # Persist
            with open(path+tile+suffix(rate)+".pkl", 'wb') as file:
                pickle.dump(rate_ds,file)
