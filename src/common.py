""" 
    This module offers the following utilities:

    - Download datasets from Carpyncho and store them locally
    - Retrieve datasets from local storage
    - Do some basic preprocessing on the datasets
    - Retrieve retrieve undersampled datasets

    Notes
    ----------
    Methods in this module will store preprocessed tiles in the 
    local filesystem:
 
    base_path/tiles/features{i}.pkl            # Features describing stars 
                                               #  in tile i
                                               
    base_path/tiles/label{i}.pkl               # Labels indicating if starts 
                                               #  in tile i are RR-Lyrae
                                               
    base_path/tiles/undersampled_indexes{i}_{rate}     # Randomly generated 
                                                       # indexes used for
                                                       # undersampling
"""

import carpyncho
import pandas as pd
from PyAstronomy.pyasl import foldAt
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn import metrics
import pickle
from sklearn.model_selection import StratifiedKFold
import random

class CarpynchoWrapper:
	"""
	This class provides methods to easily handle the datasets of variable 
	stars published by Carpyncho [1] [2].

	The datasets are available online and can be downloaed using the
	Carpyncho client. This class offers methods to store them locally,
	do some basic preprocessing and retrieve them afterwards.
	
	
	Each datasets corresponds to a tile from the VVV survey [3] and it 
	is identified by an id such as "b234".
	Each row of a dataset describes a variable star. A description of 
	how these datasets were extracted from raw telescope measurements 
	can be found in [4].
	
	Parameters
    ----------
    base_path: Filesystem path used to store and retrieve datasets in 
    pickle format.
	
	Attributes
    ----------
    base_path: Filesystem path used to store and retrieve datasets in 
    pickle format.
    
    client: Carpyncho client used to download datasets from the web.
    
	References
    ----------
    .. [1] https://carpyncho-py.readthedocs.io/en/latest/
    .. [2] Cabral et al "Carpyncho: VVV Catalog browser toolkit"
    .. [3] Minitti et al, "Vista variables in the via lactea (vvv): The 
		   public eso near-ir variability survey of the milky way" 
    .. [4] Cabral et al, "Automatic Catalog of RRLyrae from ∼ 14 million 
           VVV Light Curves"
	"""

	def __init__(self,base_path):
		self.client      = carpyncho.Carpyncho()
		self.base_path   = base_path   
		
	def preprocess_tile(self,tile):
		""" 
		Given a raw tile downloaded from Carpyncho, where each row is a variable star and each column
		is an atribute, filter start that are 'inconvenient' and construct RRL or no-RRL labels.
		
		Parameters
		----------  
		tile: string indicating the tile id.  Example: "b234".
		
		Returns
		----------  
		A tuple X,y. X is a panda dataset describing the atributes of each variable start. y is a vector
		of labels that indicates if each star described in X is of type RR-Lyrae or not 
		"""
		
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
		
	def persist_tile(self,tile):
		""" 
		Download a tile from carpyncho, filter sources and persist it in the local filesystem.
		
		Parameters
		----------  
		tile: string indicating the tile id.  Example: "b234".
		
		References
		----------
		Carpyncho docs: https://carpyncho-py.readthedocs.io/en/latest/
		"""
		df = self.client.get_catalog(tile, "features")
		train_features , train_labels = self.preprocess_tile(df)
		train_features.to_pickle(self.base_path+"/tiles/features"+tile+".pkl")  
		train_labels.to_pickle(self.base_path+"/tiles/label"+tile+".pkl")

	def persist_all_tiles(self):
		""" 
		Download, preprocess and persist all tiles 
		
		Notes
		----------
		Note: persist_all_tiles() must be invoked (only) the very first time you 
		want to use functions implemented in this file.
		"""
		for tile in self.get_all_tile_ids():
			try:
				self.persist_tile(tile)
			except:
				print("Tile "+tile+" failed to be persisted")
	
			
	def retrieve_tile(self,tile,rate="full"):
		""" 
		Retrieve a previously persisted, preprocessed tile.
		
		Notes
		----------
		This method is used throughout this project in order to retrieve
		a tile that has already been preprocessed and stored in the local
		filesystem.
		
		Parameters
		----------  
		tile: string indicating the tile id.  Example: "b234".
		rate: string or integer indicating the undersampling proportion desired.
			e.g: "full": No undersampling, "1:1000": Undersample non-RRL class
			so that there is 1 RRL for each 1000 no-RRLs.
		
		
		Returns
		----------
		A tuple X,y. X is a panda dataset describing the atributes of each variable start. y is a vector
		of labels that indicates if each star described in X is of type RR-Lyrae or not 
		"""
		
		features = pd.read_pickle(self.base_path+"/tiles/features"+ tile + ".pkl")
		labels   = pd.read_pickle(self.base_path+"/tiles/label"+ tile + ".pkl")
	
		if (rate!="full"):
			with open(self.base_path+"/tiles/undersampled_indexes"+tile+CarpynchoWrapper.suffix(rate)+".pkl", 'rb') as i_file:
				indexes = pickle.load(i_file)
			features = features.loc[indexes,]
			labels   = labels[indexes]
			
		return features, labels


	def get_supported_rates(self,tile):
		""" 
		Get all undersampling rates supported
		"""
		X,y = self.retrieve_tile(tile)
		max_rate = len(y)/sum(y)
		return([x for x in [1,10,100,150,200,250,350,425,500,750,1000,1250,1500,1750,2000,2250] if x<max_rate])

	def generate_balanced_subtiles(self,virtual=False,nsplits=20):
		""" 
		Randomly generate indexes for undersampled tiles
		"""
		ts = self.get_all_tile_ids()
		path = self.base_path+"/tiles/undersampled_indexes"

		for tile in ts:
			X,y   = self.retrieve_tile(tile)
			rrl_indexes = y[y==1].index.values # Indexes corresponding to RRLS
			rrl_n = len(rrl_indexes)           # Number of RRLS in this tile
			
			norrl_indexes = y[y==0].index.values
			random.shuffle(norrl_indexes)
			
			for rate in self.get_supported_rates(tile):
				unk_n       = rrl_n * rate   # Expected number of unknown sources in the undersampled DS with rate 'rate'
				rate_ds = rrl_indexes.tolist() + norrl_indexes[0:unk_n].tolist()
				
				# Persist
				with open(path+tile+CarpynchoWrapper.suffix(rate)+".pkl", 'wb') as file:
					pickle.dump(rate_ds,file)
    
	def get_all_tile_ids(self):
		"""
		Get the ids of all valid tiles in 'Carpyncho'
		"""
		return [ x for x in list(self.client.list_tiles()) if not(x in CarpynchoWrapper.get_invalid_tile_ids())] # All existing tiles 
	
	@staticmethod
	def get_invalid_tile_ids():
		"""
		Get the ids of all invalid tiles in 'Carpyncho'. Invalid tiles are
		filtered in all calculations.
		"""
		return ["b356","others"]
	
	@staticmethod
	def suffix(s):
		""" 
		Get the suffix string used to identify undersampled tiles
		"""
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

		
