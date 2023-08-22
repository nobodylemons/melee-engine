import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

import nvidia.cudnn
# print(nvidia.cudnn.__file__)

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


from tensorflow.python.client import device_lib

# def get_available_gpus():

# print(device_lib.list_local_devices())



import json
import os
import sys
from slippi import Game
import random

# for i in range(1000, 2000):
# work/training_data_csv/(...) Fox vs Fox [BF] Game_20200122T221201.csv
# files = os.listdir('training_data')
# random.shuffle(files)
# for filename in files:
#     game = Game(os.path.join('training_data', filename))
    

#     for i in range(1000):
#         for p in game.frames[1111].ports:
#             if p is None:
#                 continue
#             print(str(p.leader.pre.buttons.physical).split('.')[1].split('|'))
#             # p1 = game.frames[1111].ports[2].leader.pre
#         # print(dir(p0.buttons.logical))
#         # print(p1)
#         break
#     break


from lib.slp_to_csv import processFile
import multiprocess
import os
from lib.vars import *
# multiprocess.set_start_method('fork')

# print("Number of cpu : ", multiprocess.cpu_count())


myindex = 0
unprocessedFiles = []
for filename in os.listdir(slp_dir):
    if not 'slp' in filename:
        continue
    if filename.count("Fox") < 2:
        continue
    unprocessedFiles.append(filename)
    # if myindex > 25:
    #     break
    myindex += 1


from multiprocess import Pool
import tqdm
    
# pool = Pool(processes=12)
# for _ in tqdm.tqdm(pool.imap_unordered(processFile, unprocessedFiles), total=len(unprocessedFiles)):
#     pass


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# tf.config.threading.set_inter_op_parallelism_threads(64) 
# tf.config.threading.set_intra_op_parallelism_threads(64)
# tf.config.set_soft_device_placement(True)
# tf.device('/cpu:0')



import pandas as pd
import pickle
import random
from lib.data_gen import csvsToSubset

csvfiles = []
i = 0
for filename in os.listdir(data_dir):
    # if not 'csv' in filename:
    #     continue
    csvfiles.append(filename)
    i +=1
    if i%10 ==0:
        print(i)
        break
    # if i > 1:
    #     break
        
# random.shuffle(csvfiles)

# ohe = None
# if os.path.exists('encoder'):
with open('encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

print(len(ohe.get_feature_names_out()))


dfs = csvsToSubset(csvfiles[:1])


# print(list(dfs[0].columns))
# print(len(ohe.get_feature_names_out()))


from lib.data_gen import encode


sequence_len = 3
print(len(list(ohe.get_feature_names_out())))
Xshape, yshape, x_cols = encode(dfs, sequence_len, ohe)