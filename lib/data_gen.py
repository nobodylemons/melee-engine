import os
from lib.vars import *
from sklearn.preprocessing import OneHotEncoder
import pickle
from pandas import SparseDtype

def csvsToSubset(csvfiles):
    dfs = []
    for mycsv in csvfiles:
        if not 'csv' in mycsv:
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, mycsv))#.astype(float) #.fillna(-1) #.iloc[::10, :]
        dfs.append(df)
    # dfs = byHDF(dfs) 
    # dfs = pd.concat(dfs, axis=0).fillna(0)
    return dfs


def to_sequence(X_df, y, sequence_len):

    xs = []
    for i in range( len(X_df)-sequence_len):
        # if i % 10000 == 0:
        #     print(i, len(X_df))
        xs.append(X_df.iloc[i:i+sequence_len].astype(SparseDtype(float)))
    # y = np.array(y_df.iloc[sequence_len:])
    y = y[sequence_len:]
    ret_x = np.array(xs)
    ret_y = np.ravel(y) # y.reshape((np.shape(y)[0], 1))
    return ret_x, ret_y 

def dfToXyDf(df):
    x_cols = [x for x in df.columns if 'a_' == x[:2] or 's_' == x[:2]]
    # print("BUTTONS:", [x for x in df.columns if '_buttons_logical' in x])
    # CATEGORICAL_FEATURES.extend([x for x in df.columns if '_buttons_logical' in x])
    # print(x_cols)
    X_df = df[x_cols]
    y_df = df[LABELS] 
    return X_df, y_df, x_cols
    

def encode(df, sequence_len, ohe, le):
    # dfs = pd.concat(dfs, axis=0).fillna(0)
    divisor = int(len(df)/sequence_len)*sequence_len
    X_df, y_df, x_cols = dfToXyDf(df.iloc[:divisor])
    # ohe = pickle.load("encoder.pkl", "rb")
    # print(X_df[CATEGORICAL_FEATURES].astype(str))
    transformedX = ohe.transform(X_df[CATEGORICAL_FEATURES].fillna(0).astype(str) )
    # feature_names_out = ohe.get_feature_names_out()
    transformed_df = pd.DataFrame.sparse.from_spmatrix(transformedX, columns=ohe.get_feature_names_out())
    X_df = pd.concat([X_df.reset_index().drop(columns=CATEGORICAL_FEATURES, axis=1), transformed_df.reset_index()], axis=1)
    y = le.transform(y_df.values.ravel())
    # X,y = to_sequence(X_df.iloc[:sequence_len*2], y_df.iloc[:sequence_len*2], sequence_len)
    X,y = to_sequence(X_df.fillna(0), y, sequence_len)
    #240, 60, 625
    X = X_df.values.reshape((int(divisor/sequence_len), sequence_len, 625))
    y = y.reshape((int(divisor/sequence_len), sequence_len))
    # if len(np.shape(X)) == 1:
    #     print("")
    # if np.shape(X)[-1] < 2280:
    #     print("")
    X[X=='None'] = '0' # This could cause problems later but I don't want to think about it
    return X.astype(float), y, X_df.columns



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
                 ohe=None,
                 le = None,
                 sequence_len=3):
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csvfiles = csvfiles
        self.ohe = ohe
        self.le = le
        self.sequence_len = sequence_len
        # self.pca = pca
        # self.scaler = scaler
        

    def __len__(self):
        return len(self.csvfiles)

    def __getitem__(self, index):
        'Generate one batch of data'
        # print(self.csvfiles[index])
        df = pd.read_csv(os.path.join(DATA_DIR, self.csvfiles[index]))
        rand_index = random.randint(0, len(df)-self.batch_size)
        df = df.iloc[rand_index:rand_index+self.batch_size+self.sequence_len]
        
        X, y, _ = encode(df, self.sequence_len, self.ohe, self.le)
        # if 0 in list(np.shape(X)) or 0 in list(np.shape(y)):
        #     print("WHY")
        # print("Shape:", list(np.shape(X)), list(np.shape(y)))
        if np.shape(X)[-1]<625 or 0 in list(np.shape(X)) or 0 in list(np.shape(y)):
            print("BAD FILE:", self.csvfiles[index])
            return self.X, self.y
        self.X = X
        self.y = y
        return X, y

    # def on_epoch_end(self):
    #     random.shuffle(self.csvfiles)

def get_batch(batch_size, sequence_len, csvfile, ohe, le):
    df = pd.read_csv(os.path.join(DATA_DIR, csvfile))
    
    
    if batch_size > 0:
        rand_index = random.randint(0, len(df)-batch_size)
        df = df.iloc[rand_index:rand_index+batch_size+sequence_len]        
    
    X, y, _ = encode(df, sequence_len, ohe, le)
    # if 0 in list(np.shape(X)) or 0 in list(np.shape(y)):
    #     print("WHY")
    # print("Shape:", list(np.shape(X)), list(np.shape(y)))
    # if np.shape(X)[-1]<625 or 0 in list(np.shape(X)) or 0 in list(np.shape(y)):
        # print("BAD FILE:", self.csvfiles[index])

    return X, y