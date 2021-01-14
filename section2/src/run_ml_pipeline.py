"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import sys
from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from random import sample
from sklearn.model_selection import train_test_split
import numpy as np
class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"training_data"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "./section2/out"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality
    keys = list(range(len(data)))


    
    # Here, random permutation of keys array is useful in case if we do something like 
    # a k-fold training and combining the results.
    keys = np.random.permutation(keys)
    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    split['train'] = keys[0:int(0.7*len(keys))] # 70% training data
    split['val'] = keys[int(0.7*len(keys)):int(0.9*len(keys))] # 20% validation data
    split['test'] = keys[int(0.9*len(keys)):]    
    
   
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 
    


    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.


    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    del data 

    # run training
    exp.run()

    # prep and run testing
    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

