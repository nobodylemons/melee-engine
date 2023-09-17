from __future__ import absolute_import, division, print_function, unicode_literals

import os
from tempfile import gettempdir

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model

# from clearml import Task
import pickle
import numpy as np
import tqdm

from lib.data_gen import csvsToSubset
from lib.vars import LABELS, DATA_DIR
from lib.data_gen import get_batch
from lib.time2vec import Time2Vec
from lib.positional_embedding import PositionalEmbedding

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  


# Connecting ClearML with the current process,
# from here on everything is logged automatically
# task = Task.init(project_name='Melee Engine', task_name='My Task')


# Build the tf.keras model using the Keras model subclassing API
class MyModel(Model):
    def __init__(self, head_size, num_heads, ff_dim, mlp_units, dropout, mlp_dropout , input_shape, output_shape):
        super(MyModel, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout

        self.ln0 = layers.LayerNormalization(epsilon=1e-6)

        # self.ttv = Time2Vec(kernel=625)
        

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)

        self.te_ln0 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_heads)]
        self.te_multi_head = [layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout) for _ in range(num_heads)]
        self.te_dropout0 = [layers.Dropout(self.dropout) for _ in range(num_heads)]

        self.te_ln1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_heads)]
        self.te_conv0 = [layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu") for _ in range(num_heads)]
        self.te_dropout1 = [layers.Dropout(self.dropout) for _ in range(num_heads)]
        self.te_conv1 = [layers.Conv1D(filters=input_shape[-1], kernel_size=1) for _ in range(num_heads)]

        self.pool = layers.GlobalAveragePooling1D(data_format="channels_first") # for _ in range(num_heads)]
        
        self.mlp_dense = []
        self.mlp_drop = []
        for dim in mlp_units:
            self.mlp_dense.append(layers.Dense(dim, activation="relu"))
            self.mlp_drop.append(layers.Dropout(self.mlp_dropout))

        self.out_dense = layers.Dense(output_shape[-1], activation="softmax")

    def call_transformer_encoder(self, inputs, i):
        # Attention and Normalization 
        x = self.te_ln0[i](inputs)
        x = self.te_multi_head[i](x, x)
        x = self.te_dropout0[i](x)
    #     x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = self.te_ln1[i](res)
        x = self.te_conv0[i](x)
        x = self.te_dropout1[i](x)
        x = self.te_conv1[i](x)
        
        return x + res

    def call(self, x):
        # x = self.ln0(x)
        
        # output_list = []

        # for col in range(x.shape[-1]):
        #     print("X SHAPE:", x.shape)
        #     print("SHAPE:",col,x[:,:,col].shape)
        #     result = self.ttv(x[:,:,col])
        #     print("result shape:", result.shape)
        #     output_list.append(result)
        #     # x[:,:,col] = result
        # x = tf.stack(output_list)
        # print("NEW X SHAPE", x.shape)

        x = self.ttv(x)
        # x = tf.map_fn(self.ttv, x, back_prop=False)
        # np.apply_along_axis(self.ttv, axis=2, arr=x)
        # x = self.ttv(x)
        x = self.ln1(x)
        for i in range(self.num_heads):
            x = self.call_transformer_encoder(x, i)
        # outs = []
        # for y in ys:
        x = self.pool(x)
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense[i](x)
            x = self.mlp_drop[i](x)
        # outputs = layers.Dense(y_len, activation="sigmoid")(x)
        # outputs = layers.Dense(2, activation="linear")(x)
        # outputs = layers.Dense(370, activation="softmax")(x)
        # outs.append(outputs)
        #            layers.Dense(np.shape(ys[1])[1], activation="softmax")(layers.Dense(30, activation="relu")(x))]
        return self.out_dense(x)


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
        if i >= n and n != -1:
            break
        # if i > 1:
        #     break
    return csvfiles
            
csvfiles = get_csv_files(1)

with open('one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

dfs = csvsToSubset(csvfiles)

from lib.data_gen import encode


EPOCHS = 50
BATCH_SIZE = 128
SEQUENCE_LEN = 60

X, y, x_cols = encode(dfs[0].iloc[:SEQUENCE_LEN*5], SEQUENCE_LEN, ohe, le)

# Create an instance of the model
model = MyModel(head_size=256, 
                num_heads=1, 
                ff_dim=4, 
                mlp_units=[128], 
                dropout=.2, 
                mlp_dropout=.2,
                input_shape=X.shape, 
                output_shape=[len(le.classes_)])

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Use tf.GradientTape to train the model
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# Test the model
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# Set up summary writers to write the summaries to disk in a different logs directory
train_log_dir = os.path.join(gettempdir(), 'logs', 'gradient_tape', 'train')
test_log_dir = os.path.join(gettempdir(), 'logs', 'gradient_tape', 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Set up checkpoints manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, os.path.join(gettempdir(), 'tf_ckpts'), max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# Start training

for epoch in range(EPOCHS):
    for csv_file in tqdm.tqdm(get_csv_files()):
    # for features, labels in get_batch(batch_size, sequence_len, csvfile, ohe, le):
        features, labels = get_batch(BATCH_SIZE, SEQUENCE_LEN, csv_file, ohe, le)
        if 0 in np.shape(features) or 0 in np.shape(labels) or np.shape(features)[-1]<625:
            continue
        train_step(features, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    ckpt.step.assign_add(1)
    if int(ckpt.step) % 1 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    # for test_images, test_labels in test_ds:
    #     test_step(test_images, test_labels)
    #     with test_summary_writer.as_default():
    #         tf.summary.scalar('loss', test_loss.result(), step=epoch)
    #         tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}' #, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100))
                        #   test_loss.result(),
                        #   test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    # test_loss.reset_states()
    # test_accuracy.reset_states()