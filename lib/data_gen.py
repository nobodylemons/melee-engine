import os
from lib.vars import *
from sklearn.preprocessing import OneHotEncoder
import pickle

def csvsToSubset(csvfiles):
    dfs = []
    for mycsv in csvfiles:
        if not 'csv' in mycsv:
            continue

        df = pd.read_csv(os.path.join(data_dir, mycsv)).astype(float) #.fillna(-1) #.iloc[::10, :]
        dfs.append(df)
    # dfs = byHDF(dfs) 
    # dfs = pd.concat(dfs, axis=0).fillna(0)
    return dfs


def to_sequence(X_df, y_df, sequence_len):

    xs = []
    for i in range( len(X_df)-sequence_len):
        # if i % 10000 == 0:
        #     print(i, len(X_df))
        xs.append(np.array(X_df.iloc[i:i+sequence_len+1]))
    y = np.array(y_df.iloc[sequence_len:])
    return np.array(xs), y.reshape((np.shape(y)[0], len(y_df.columns)))

def dfToXyDf(df):
    x_cols = [x for x in df.columns if 'a_' == x[:2] or 's_' == x[:2]]
    # print("BUTTONS:", [x for x in df.columns if '_buttons_logical' in x])
    # categoricalFeatures.extend([x for x in df.columns if '_buttons_logical' in x])
    # print(x_cols)
    X_df = df[x_cols]
    y_df = df[['pred_player0_state']] #, 'pred_player1_state']]
    return X_df, y_df, x_cols
    

def encode(dfs, sequence_len, ohe):
    dfs = pd.concat(dfs, axis=0).fillna(0)
    X_df, y_df, x_cols = dfToXyDf(dfs)
    # ohe = pickle.load("encoder.pkl", "rb")
    print(X_df[categoricalFeatures].astype(str))
    transformedX = ohe.transform(X_df[categoricalFeatures].astype(str))
    
    transformed_df = pd.DataFrame(transformedX, columns=ohe.get_feature_names_out())
    X_df = pd.concat([X_df, transformed_df], axis=1).drop(columns=categoricalFeatures, axis=1)

    X,y = to_sequence(X_df.iloc[:sequence_len*2], y_df.iloc[:sequence_len*2], sequence_len)
    return np.shape(X), np.shape(y), x_cols



import numpy as np
import keras
import random
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_col, y_col,
                 batch_size,
                 shuffle=True,
                 csvfiles=[],
                 ohe=None):
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csvfiles = csvfiles
        self.ohe = ohe
        # self.pca = pca
        # self.scaler = scaler
        

    def __len__(self):
        return len(self.csvfiles)

    def __getitem__(self, index):
        'Generate one batch of data'
        df = pd.read_csv(os.path.join('meleeEvalBar', self.csvfiles[index]))
        df = df[df.next_death != -1].dropna()
        if len(df) < 4:
            return
            
        dfEven = df.iloc[::2]
        dfOdd = df.iloc[1::2].add_suffix('_odd')
        df = pd.concat([dfEven.reset_index(drop=True), 
                        dfOdd[['player/1/x_odd', 'player/1/y_odd', 'player/2/x_odd', 'player/2/y_odd']].reset_index(drop=True)],
                      axis=1)
#         print(np.shape(df), self.csvfiles[index])
        df = df.dropna().sample(self.batch_size, replace=True)
        X, y = ohe_transform(df, self.ohe)
#         X = self.scaler.transform(X)
#         X = self.pca.transform(X)
        return X, y

    def on_epoch_end(self):
        random.shuffle(self.csvfiles)