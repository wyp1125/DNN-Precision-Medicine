#  Copyright 2017 Yupeng Wang.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  This application program depends on Tensorflow v1.2 and Python3.
#  Part of its code is derived from the tutoring programs provided
#  by the TensorFlow Authors.

"""Title: DNN model to process new datasets containing categorical columns"""

#  The input files should be comma delimited (csv) and contain a header line
#  of sample names and 'label'. At least one categorical columns should be
#  provded as a command argument. The last column of the data is 'label', 
#  coded by 0,1,...,#classes. All other columns should be feature data.
#  Sample command:
#  Author: Yupeng Wang. ywangbusiness@gmail.com.
#  Sample command: python3 dnn_model_pandas_cat.py train.csv test.csv s7,s8,s9 3 10,20,10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
import pandas as pd

if len(sys.argv)<7:
  print("Usage:python3 dnn_model_pandas_cat.py training_file testing_file categorical_columns(comma delimited) #classes layer_units(comma delimited) #steps")
  quit()

#process header
with open(sys.argv[1],'r') as fl:
  line=fl.readline().strip('\n')
header=line.split(',')
n_col=len(header)
features=[]
for i in range(n_col-1):
  features.append(header[i])
LABEL_COLUMN=header[n_col-1]
#process arguments
nsteps=int(sys.argv[6])
n_cls=int(sys.argv[4])
h_units=[]
lt=sys.argv[5].split(',')
for w in lt:
  h_units.append(int(w))
print("DNN units:")
print(h_units)
#read in training and testing data
df_train = pd.read_csv(sys.argv[1], names=header, skiprows=1, skipinitialspace=True)
df_test = pd.read_csv(sys.argv[2], names=header, skiprows=1, skipinitialspace=True)
#assign feature types for the DNN model
CATEGORICAL_COLUMNS=[]
categorical_ind=[]
CONTINUOUS_COLUMNS=[]
continuous_ind=[]
tt=sys.argv[3].split(',')
for w in tt:
  CATEGORICAL_COLUMNS.append(w)
  categorical_ind.append(tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket(
    w, hash_bucket_size=1000),dimension=8))
for i in range(n_col-1):
  if header[i] not in CATEGORICAL_COLUMNS:
    CONTINUOUS_COLUMNS.append(header[i])
    continuous_ind.append(tf.contrib.layers.real_valued_column(header[i]))
deep_columns=[]
for s in categorical_ind:
  deep_columns.append(s)
for s in continuous_ind:
  deep_columns.append(s)
#convert input data into tensors
def input_fn(df):
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  label = tf.constant(df[LABEL_COLUMN].values)
  return feature_cols, label
#train the DNN model and evaluate its performance
def main(unused_argv):
  m = tf.contrib.learn.DNNClassifier(
      feature_columns=deep_columns,
      hidden_units=h_units,
      n_classes=n_cls)
  m.fit(input_fn=lambda: input_fn(df_train), steps=nsteps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  print(results)

if __name__ == "__main__":
  tf.app.run()
