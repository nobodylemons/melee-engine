import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pickle
import copy
from joblib import dump
from lib.vars import SLP_DIR, DATA_DIR, CATEGORICAL_FEATURES, LABELS
import numpy as np


# SLP_DIR = '/home/robert/slippi_data/Slippi_Public_Dataset_v3'
# DATA_DIR = '/root/training_data_csv'

['s_stage', 'index', 'pred_player0_state', 's_player0_state',
       's_player0_airborne', 's_player0_character', 's_player0_combo_count',
       's_player0_damage', 's_player0_direction', 's_player0_ground',
       's_player0_hit_stun', 's_player0_jumps', 's_player0_l_cancel',
       's_player0_last_attack_landed', 's_player0_last_hit_by',
       's_player0_position_x', 's_player0_position_y', 's_player0_shield',
       's_player0_state_age', 's_player0_stocks',
       'a_player0_buttons_physical_Y', 'a_player0_buttons_physical_R',
       'a_player0_buttons_physical_B', 'a_player0_buttons_physical_L',
       'a_player0_buttons_physical_Z', 'a_player0_buttons_physical_A',
       'a_player0_buttons_physical_X', 'a_player0_buttons_physical_START',
       'a_player0_buttons_physical_DPAD_UP',
       'a_player0_buttons_physical_DPAD_RIGHT',
       'a_player0_buttons_physical_DPAD_DOWN',
       'a_player0_buttons_physical_DPAD_LEFT',
       'a_player0_buttons_physical_NONE', 'a_player0_cstick_x',
       'a_player0_cstick_y', 'a_player0_joystick_x', 'a_player0_joystick_y',
       'a_player0_triggers_logical', 'pred_player1_state', 's_player1_state',
       's_player1_airborne', 's_player1_character', 's_player1_combo_count',
       's_player1_damage', 's_player1_direction', 's_player1_ground',
       's_player1_hit_stun', 's_player1_jumps', 's_player1_l_cancel',
       's_player1_last_attack_landed', 's_player1_last_hit_by',
       's_player1_position_x', 's_player1_position_y', 's_player1_shield',
       's_player1_state_age', 's_player1_stocks',
       'a_player1_buttons_physical_Y', 'a_player1_buttons_physical_R',
       'a_player1_buttons_physical_B', 'a_player1_buttons_physical_L',
       'a_player1_buttons_physical_Z', 'a_player1_buttons_physical_A',
       'a_player1_buttons_physical_X', 'a_player1_buttons_physical_START',
       'a_player1_buttons_physical_DPAD_UP',
       'a_player1_buttons_physical_DPAD_RIGHT',
       'a_player1_buttons_physical_DPAD_DOWN',
       'a_player1_buttons_physical_DPAD_LEFT',
       'a_player1_buttons_physical_NONE', 'a_player1_cstick_x',
       'a_player1_cstick_y', 'a_player1_joystick_x', 'a_player1_joystick_y',
       'a_player1_triggers_logical']

CATEGORICAL_FEATURES = [ 's_stage',

                       's_player0_state',
                    #    's_player0_character',
                    #    's_player0_flags',
                       's_player0_ground',
                    #    'a_player0_buttons_logical',
                       # 'a_player0_state',

                       's_player1_state',
                    #    's_player1_character',
                    #    's_player1_flags',
                       's_player1_ground',
                    #    'a_player1_buttons_logical',
                       # 'a_player1_state'
                        "s_player0_last_attack_landed",
                         "s_player1_last_attack_landed",
                         "s_player0_last_hit_by",
                         "s_player1_last_hit_by"
                      ]


POSITIONAL_FEATURES = ['s_player0_position_x',
                       's_player0_position_y',
                       's_player1_position_x',
                       's_player1_position_y']

FEATURES = []
FEATURES.extend(CATEGORICAL_FEATURES)
FEATURES.extend(POSITIONAL_FEATURES)

cols = copy.deepcopy(CATEGORICAL_FEATURES)

ohe_path = 'onehotencoder.joblib'
le_path = 'labelencoder.joblib'

if not (os.path.exists(ohe_path) and os.path.exists(le_path)):
# if True:
    i = 0
    dfs = []
    for filename in os.listdir(DATA_DIR):
        # if i > 100:
        #     break
        if not 'csv' in filename:
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, filename), dtype=str) #.astype(float)
        # print(df.columns)
        df = df[cols].astype(str).drop_duplicates()
        # print(df)
        dfs.append(df)
        print(filename)
        i += 1
    # dfs = csvsToSubset(filenames)
    dfs = pd.concat(dfs, axis=0).fillna(0)
    les = {}


    les["s_stage"] = LabelEncoder()
    les["s_stage"].fit(dfs[["s_stage"]])


    for k in ["s_player0_state", 's_player0_ground', "s_player0_last_attack_landed", "s_player0_last_hit_by"]:
        ohe_key = k.replace("_player0", "")
        les[ohe_key] = LabelEncoder()
        df0 = dfs[[k]].rename(columns={k: ohe_key})
        df1 = dfs[[k.replace("0", "1")]].rename(columns={k.replace("0", "1"): ohe_key})
        state_df = np.concatenate([df0[ohe_key].values, df1[ohe_key].values]).astype(str)
        les[ohe_key].fit(state_df)
        # if k == "s_player0_ground":
        #     print("")
    dump(les, le_path)

from joblib import load

les = load(le_path)


import os
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd
import numpy as np
import random
import warnings



def csvsToSubset(csvfiles, sequence_len=0, batch_size=0):
    dfs = []
    dtype_filename = os.path.join(DATA_DIR, 'dtype_file.csv')
    dtype_df = pd.read_csv(dtype_filename, index_col=0).to_dict()['0']

    for mycsv in csvfiles:
        if not 'csv' in mycsv:
            continue
        if 'dtype_file' in mycsv:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(os.path.join(DATA_DIR, mycsv), dtype=dtype_df) #.astype(float) #.fillna(-1) #.iloc[::10, :]
            if len(df) < sequence_len * batch_size:
                continue
            dfs.append(df)

    return dfs


import numpy as np
import random

def to_sequence(features_dict, y_dict, sequence_len):
    ret_dict = {}
    rand_ind = random.randint(0, sequence_len-1)  # To add some randomness to the starting index

    # Pre-compute the range to avoid redundant calculations inside the loop
    sequence_range = range(rand_ind, len(y_dict["s_player0_position_x"]) - sequence_len)

    for k, v in features_dict.items():
        # Use list comprehension for memory efficiency
        xs = [np.array(v[i:i+sequence_len]) for i in sequence_range]

        # Convert list to numpy array only once after the list is fully formed
        xs = np.array(xs)

        # Reshape efficiently
        if xs.ndim == 3 and xs.shape[-1] == 1:
            xs = xs.reshape(xs.shape[0], xs.shape[1])

        ret_dict[k] = xs
    ret_y = {}
    for k, v in y_dict.items():
        ys = [np.array(v[i+1:i+sequence_len+1]) for i in sequence_range]
        ret_y[k] = np.array(ys)
    return ret_dict, ret_y

def encode(df, sequence_len, les):
    features_dict = {}

    # Optimize label encoding process
    for cat in CATEGORICAL_FEATURES:
        ohe_key = cat.replace("player0_", "").replace("player1_", "")
        features_dict[cat] = les[ohe_key].transform(df[cat].ravel())

    # Direct assignment for positional features
    for pos in POSITIONAL_FEATURES:
        features_dict[pos] = df[pos].values

    # Handling 'other_features'
    other_features = [col for col in df.columns if "a_player" in col]
    features_dict["other_features"] = df[other_features].values  # Consider if '.values' is needed based on later usage

    y_dict = {
            "s_player0_state": les["s_state"].transform(df["s_player0_state"].ravel()),
            "s_player1_state": les["s_state"].transform(df["s_player1_state"].ravel()),
            "s_player0_position_x": df["s_player0_position_x"],
            "s_player0_position_y": df["s_player0_position_y"],
            "s_player1_position_x": df["s_player1_position_x"],
            "s_player1_position_y": df["s_player1_position_y"],
              }

    # y = features_dict['s_player0_state'] #.to_numpy()
    # y = df[["s_player0_position_x"]].to_numpy()
    features_dict, y_dict = to_sequence(features_dict, y_dict, sequence_len)

    return features_dict, y_dict



def get_csv_files(n=-1):

    csvfiles = []
    i = 0
    for filename in os.listdir(DATA_DIR):
        if not 'csv' in filename:
            continue
        csvfiles.append(filename)
        i +=1
        # if i%10 ==0:
        #     print(i)
        # if i >= n and n != -1:
        #     break
        # if i > 1:
        #     break
    return csvfiles

csvfiles = get_csv_files()

dfs = csvsToSubset(csvfiles[:2])

sequence_len = 10
batch_size = 32
features_dict, y_dict  = encode(dfs[0], sequence_len, les)



from tqdm import tqdm  # Import tqdm
import torch

csvfiles = get_csv_files()
dfs = csvsToSubset(csvfiles[:int(len(csvfiles)/4)], sequence_len, batch_size)
# dfs = csvsToSubset(csvfiles, sequence_len, batch_size)

import numpy as np
import torch

def get_batches(features_dict, y_dict, batch_size, device):
    # Convert data to PyTorch tensors if not already
    for k, v in features_dict.items():
        if isinstance(features_dict[k], torch.Tensor):
            continue
        if k in CATEGORICAL_FEATURES:
            features_dict[k] = torch.tensor(v).to(device).to(torch.long)
        else:
            features_dict[k] = torch.tensor(v.astype(np.float32)).to(device)
        # print(k, v.shape)
    # features_dict = {k: torch.tensor(v).to(device) if not isinstance(v, torch.Tensor) else v
    #         for k, v in data.items()}
    for k, y in y_dict.items():
        y_dict[k] = torch.tensor(y).to(device) if not isinstance(y, torch.Tensor) else y

    batches, y_batches = [], []
    total_length = features_dict[list(features_dict.keys())[0]].size(0)

    for i in range(0, total_length, batch_size):
        batch = {}
        for k, v in features_dict.items():
            batch_end = min(i+batch_size, total_length)
            batch[k] = v[i:batch_end]
        batches.append(batch)
        y_batch = {}
        for k, y in y_dict.items():
            y_batch[k] = y[i:batch_end] #.to(torch.float)
        y_batches.append(y_batch)
    return batches, y_batches
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MixedFeatureTransformer(nn.Module):
    def __init__(self, features_dict, batch_size, sequence_len, embed_size, num_heads, ff_hid_dim, num_layers, les, device):
        super(MixedFeatureTransformer, self).__init__()
        self.dev = device
        self.batch_size=batch_size
        self.sequence_len = sequence_len
        self.les = les

        self.emb_coords0 = nn.Linear(in_features=2, out_features=50).to(self.dev)
        self.emb_coords1 = nn.Linear(in_features=2, out_features=50).to(self.dev)

        stage_len = len(les["s_stage"].classes_)
        self.emb_stage = nn.Embedding(num_embeddings=stage_len, embedding_dim=10).to(self.dev)

        state_len = len(les["s_state"].classes_)
        self.emb_states0 = nn.Embedding(num_embeddings=state_len, embedding_dim=50).to(self.dev)
        self.emb_states1 = nn.Embedding(num_embeddings=state_len, embedding_dim=50).to(self.dev)

        ground_len = len(les["s_ground"].classes_)
        self.emb_ground0 = nn.Embedding(num_embeddings=ground_len, embedding_dim=25).to(self.dev)
        self.emb_ground1 = nn.Embedding(num_embeddings=ground_len, embedding_dim=25).to(self.dev)

        last_hit_by_len = len(les["s_last_hit_by"].classes_)
        self.emb_last_hit_by0 = nn.Embedding(num_embeddings=last_hit_by_len, embedding_dim=25).to(self.dev)
        self.emb_last_hit_by1 = nn.Embedding(num_embeddings=last_hit_by_len, embedding_dim=25).to(self.dev)

        last_attack_landed_len = len(les["s_last_attack_landed"].classes_)
        self.emb_last_attack_landed0 = nn.Embedding(num_embeddings=last_attack_landed_len, embedding_dim=25).to(self.dev)
        self.emb_last_attack_landed1 = nn.Embedding(num_embeddings=last_attack_landed_len, embedding_dim=25).to(self.dev)


        self.big_linear = nn.Linear(in_features=np.shape(features_dict["other_features"])[-1], out_features=256).to(self.dev)

        transformer_size = 656
        self.pos_encoder = PositionalEncoding(d_model=transformer_size, max_len=self.sequence_len).to(self.dev)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=transformer_size, nhead=num_heads, dim_feedforward=ff_hid_dim),
            num_layers=num_layers
        ).to(self.dev)

        # Output layer
        self.p0_state_out_layer = nn.Linear(transformer_size, state_len).to(self.dev)
        self.p1_state_out_layer = nn.Linear(transformer_size, state_len).to(self.dev)
        self.p0_pos_x_layer = nn.Linear(transformer_size, 1).to(self.dev)
        self.p0_pos_y_layer = nn.Linear(transformer_size, 1).to(self.dev)
        self.p1_pos_x_layer = nn.Linear(transformer_size, 1).to(self.dev)
        self.p1_pos_y_layer = nn.Linear(transformer_size, 1).to(self.dev)

    def forward(self, features_dict):
        embeddings = []


        emb = self.emb_states0(features_dict["s_stage"])
        embeddings.append(emb)

        emb = self.emb_states0(features_dict["s_player0_state"])
        embeddings.append(emb)

        emb = self.emb_states1(features_dict["s_player1_state"])
        embeddings.append(emb)

        emb = self.emb_ground0(features_dict["s_player0_ground"])
        embeddings.append(emb)

        emb = self.emb_ground1(features_dict["s_player1_ground"])
        embeddings.append(emb)

        emb = self.emb_last_hit_by0(features_dict["s_player0_last_hit_by"])
        embeddings.append(emb)

        emb = self.emb_last_hit_by1(features_dict["s_player1_last_hit_by"])
        embeddings.append(emb)

        emb = self.emb_last_attack_landed0(features_dict["s_player0_last_attack_landed"])
        embeddings.append(emb)

        emb = self.emb_last_attack_landed1(features_dict["s_player1_last_attack_landed"])
        embeddings.append(emb)

        x = features_dict['s_player0_position_x']
        y = features_dict['s_player0_position_y']
        stack = torch.stack((x,y), dim=-1)
        # print("Stack shape:", stack.shape)
        emb = self.emb_coords0(stack)
        embeddings.append(emb)

        x = features_dict['s_player1_position_x']
        y = features_dict['s_player1_position_y']
        emb = self.emb_coords1(torch.stack((x,y), dim=-1))
        embeddings.append(emb)

        emb = self.big_linear(features_dict['other_features'])
        embeddings.append(emb)
        total_embedding = torch.cat(embeddings,
                                    dim=-1)
        total_embedding = total_embedding.permute(1, 0, 2)

        total_embedding = self.pos_encoder(total_embedding)
        # print("Pre-transformer:", total_embedding.shape)


        transformer_output = self.transformer(total_embedding)
        # output = transformer_output.mean(dim=1)

        # output = self.fc_out(transformer_output)
        p0_state_out = self.p0_state_out_layer(transformer_output)
        p1_state_out = self.p1_state_out_layer(transformer_output)
        p0_pos_x_out = self.p0_pos_x_layer(transformer_output)
        p0_pos_y_out = self.p0_pos_y_layer(transformer_output)
        p1_pos_x_out = self.p1_pos_x_layer(transformer_output)
        p1_pos_y_out = self.p1_pos_y_layer(transformer_output)

        return {
            'p0_state': p0_state_out,
            'p1_state': p1_state_out,
            'p0_pos_x': p0_pos_x_out.reshape((p0_pos_x_out.shape[1], p0_pos_x_out.shape[0])),
            'p0_pos_y': p0_pos_y_out.reshape((p0_pos_x_out.shape[1], p0_pos_x_out.shape[0])),
            'p1_pos_x': p1_pos_x_out.reshape((p0_pos_x_out.shape[1], p0_pos_x_out.shape[0])),
            'p1_pos_y': p1_pos_y_out.reshape((p0_pos_x_out.shape[1], p0_pos_x_out.shape[0]))
        }



import torch.nn.functional as F


# Hyperparameters
embed_size = 512  # Size of embedding vector
num_heads = 4  # Number of attention heads
ff_hid_dim = 512  # Hidden layer size in FFN
num_layers = 2  # Number of encoder and decoder layers
lr = 0.0001  # Learning rate

# Instantiate model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MixedFeatureTransformer(features_dict, batch_size, sequence_len, embed_size, num_heads, ff_hid_dim, num_layers, les, device)

criterion_state = nn.CrossEntropyLoss()
criterion_position = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


model.to(device)




def calculate_accuracy(y_pred, true_labels):
    # Apply softmax to convert to probability scores if needed
    softmax = nn.Softmax(dim=-2)
    probabilities = softmax(y_pred)

    # Get predicted labels
    # print("Y pred:", y_pred.shape)
    _, predicted_labels = torch.max(probabilities, dim=-2)

    # If y_true is also one-hot encoded, convert it to indices
    # _, true_labels = torch.max(y_true, dim=-1)

    # Check if the predicted labels match the true labels
    # print(predicted_labels.shape)
    # print(true_labels.shape)
    correct_predictions = (predicted_labels == true_labels).float()

    # Calculate accuracy
    accuracy = correct_predictions.sum() / correct_predictions.numel()
    return accuracy



import matplotlib.pyplot as plt
from IPython.display import clear_output

def live_plot(losses, accuracies, figsize=(12,5)):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plotting losses
    # for loss_dict in losses:
    for key, value in losses.items():
        ax1.plot(value, label=key)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plotting accuracies
    # for accuracy_dict in accuracies:
    for key, value in accuracies.items():
        ax2.plot(value[-100:] if len(value) > 100 else value, label=key)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.set_ylim([None, 1])
    ax2.legend()

    # plt.show()
    plt.savefig("training.png", bbox_inches='tight')
    plt.close()
    




from tqdm import tqdm
from random import shuffle


epochs = 100
losses = {"loss_state_p0":[],
                "loss_state_p1":[],
                "loss_pos_p0_x":[],
                "loss_pos_p0_y":[],
                "loss_pos_p1_x":[],
                "loss_pos_p1_y":[],
                }
accuracies = {"s_player0_state":[],
                    "s_player1_state":[]}
# Iterate over epochs with a progress bar
for i in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    i = 0
    shuffle(dfs)
    for df in tqdm(dfs):  # Add tqdm here for progress bar
        features_dict, y_dict = encode(df, sequence_len, les)


        # Wrap your inner loop with tqdm for a nested progress bar
        # for features_dict, y in tqdm(zip(features_dicts, ys), desc=f"Epoch {i+1}/{epochs}", unit="batch", leave=False):
        loss_dict = {"loss_state_p0":0,
                     "loss_state_p1":0,
                     "loss_pos_p0_x":0,
                     "loss_pos_p0_y":0,
                     "loss_pos_p1_x":0,
                     "loss_pos_p1_y":0,
                     }
        accuracy_dict = {"s_player0_state":0,
                         "s_player1_state":0}

        div = 0
        batches, y_batches = get_batches(features_dict, y_dict, batch_size, device)

        for batch, y_batch in zip(batches, y_batches):

            # batch = {k: v.float() for k, v in batch.items()}
            # y_batch = y_batch.permute(1, 0, 2)
            outputs = model(batch) #.permute( 1,2,0)
            # y_batch = y_batch.permute(0,1)

            accuracy_dict["s_player0_state"] += calculate_accuracy(outputs['p0_state'].permute(0,2,1), y_batch['s_player0_state'].permute(1,0)).cpu().detach().numpy()
            accuracy_dict["s_player1_state"] += calculate_accuracy(outputs['p1_state'].permute(0,2,1), y_batch['s_player1_state'].permute(1,0)).cpu().detach().numpy()

            # Calculate loss for each output
            # print(outputs['p0_state'].shape, y_batch['s_player0_state'].shape)
            loss_state_p0 = criterion_state(outputs['p0_state'].permute(0,2,1), y_batch['s_player0_state'].permute(1,0))
            loss_state_p1 = criterion_state(outputs['p1_state'].permute(0,2,1), y_batch['s_player1_state'].permute(1,0))
            loss_pos_p0_x = criterion_position(outputs['p0_pos_x'], y_batch['s_player0_position_x'].float())
            loss_pos_p0_y = criterion_position(outputs['p0_pos_y'], y_batch['s_player0_position_y'].float())
            loss_pos_p1_x = criterion_position(outputs['p1_pos_x'], y_batch['s_player1_position_x'].float())
            loss_pos_p1_y = criterion_position(outputs['p1_pos_y'], y_batch['s_player1_position_y'].float())

            loss_dict["loss_state_p0"] += loss_state_p0.cpu().detach().numpy()
            loss_dict["loss_state_p1"] += loss_state_p1.cpu().detach().numpy()
            loss_dict["loss_pos_p0_x"] += loss_pos_p0_x.cpu().detach().numpy()
            loss_dict["loss_pos_p0_y"] += loss_pos_p0_y.cpu().detach().numpy()
            loss_dict["loss_pos_p1_x"] += loss_pos_p1_x.cpu().detach().numpy()
            loss_dict["loss_pos_p1_y"] += loss_pos_p1_y.cpu().detach().numpy()



            # Combine losses
            total_loss = loss_state_p0 + loss_state_p1 + loss_pos_p0_x + loss_pos_p0_y + loss_pos_p1_x + loss_pos_p1_y


            div += 1

            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Calculate average loss and accuracy
        for k, v in loss_dict.items():
            loss_dict[k] = v / div if div > 0 else 0

        for k, v in accuracy_dict.items():
            accuracy_dict[k] = v / div if div > 0 else 0
        # Append to lists

        for k, v in losses.items():
            losses[k].append(loss_dict[k])

        for k, v in accuracies.items():
            accuracies[k].append(accuracy_dict[k])

        # Update plot
        live_plot(losses, accuracies)
        # if div > 0:
        #     print("Loss:", average_loss / div, "Accuracy:", acc / div)




