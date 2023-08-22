from lib.data_gen import dfToXyDf, csvsToSubset
from lib.vars import *
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder()

filenames = []
dfs = []
for filename in os.listdir(data_dir):
    if not 'csv' in filename:
        continue
    # print(filename)
    # filenames.append(filename)
    df = pd.read_csv(os.path.join(data_dir, filename)).astype(int) 
    # df = df[categoricalFeatures].drop_duplicates()
    dfs.append(df)
    print(filename)

# dfs = csvsToSubset(filenames)

dfs = pd.concat(dfs, axis=0).fillna(0)
# X_df, y_df, x_cols = dfToXyDf(dfs)
ohe.fit(dfs[categoricalFeatures].astype(str))


import pickle
with open("encoder.pkl", "wb") as f: 
    pickle.dump(ohe, f)