from lib.data_gen import dfToXyDf, csvsToSubset
from lib.vars import SLP_DIR, DATA_DIR, CATEGORICAL_FEATURES, LABELS
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pickle
import copy

cols = copy.deepcopy(CATEGORICAL_FEATURES)
cols.extend(LABELS)

i = 0
dfs = []
for filename in os.listdir(DATA_DIR):
    # if i > 1:
    #     break
    if not 'csv' in filename:
        continue
    df = pd.read_csv(os.path.join(DATA_DIR, filename)) #.astype(float) 

    df = df[cols].drop_duplicates()
    # print(df)
    dfs.append(df)
    print(filename)
    i += 1
print("COLUMNS", cols)
# dfs = csvsToSubset(filenames)
dfs = pd.concat(dfs, axis=0).fillna(0)
# # X_df, y_df, x_cols = dfToXyDf(dfs)


ohe = OneHotEncoder()
ohe.fit(dfs[CATEGORICAL_FEATURES].astype(str))
with open("one_hot_encoder.pkl", "wb") as f: 
    pickle.dump({"encoder":ohe, "features": CATEGORICAL_FEATURES}, f)

le = LabelEncoder()
le.fit(dfs[LABELS[0]].astype(str))

with open("label_encoder.pkl", "wb") as f: 
    pickle.dump({"encoder":le, "labels": LABELS} , f)