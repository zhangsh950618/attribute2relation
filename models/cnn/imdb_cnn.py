'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Dot
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils import to_categorical
import os
from keras.models import load_model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="6"
# set parameters:
max_features = 5000
maxlen = 200
batch_size = 1024
embedding_dims = 120
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 100
DIRNAME = './'

embedding_matrix = np.load(os.path.join(DIRNAME, "embedding_matrix.npy"))
x_train_h_description = np.load(os.path.join(DIRNAME,"train_h_description.npy"))
x_train_t_description = np.load(os.path.join(DIRNAME,"train_t_description.npy"))
x_train_r = np.load(os.path.join(DIRNAME,"train_r.npy"))
y_train = np.load(os.path.join(DIRNAME,"train_label.npy"))
# y_train = to_categorical(np.asarray(y_train), num_classes = 2)


# x_test_h_description = np.load(os.path.join(DIRNAME,"test_h_description.npy"))
# x_test_t_description = np.load(os.path.join(DIRNAME,"test_t_description.npy"))
# x_test_r = np.load(os.path.join(DIRNAME,"test_r.npy"))
# y_test = np.load(os.path.join(DIRNAME,"test_label.npy"))
# y_test = to_categorical(np.asarray(y_test), num_classes = 2)

print('Pad sequences (samples x time)')
x_train_h_description = sequence.pad_sequences(x_train_h_description, maxlen=maxlen)
x_train_t_description = sequence.pad_sequences(x_train_t_description, maxlen=maxlen)
np.save("train_h_description_pad.npy", x_train_h_description)
np.save("train_t_description_pad.npy", x_train_t_description)
x_train_r = sequence.pad_sequences(x_train_r, maxlen=4)

print(type(x_train_h_description))
print(x_train_h_description.shape)
# x_test_h_description = sequence.pad_sequences(x_test_h_description, maxlen=maxlen)
# x_test_t_description = sequence.pad_sequences(x_test_t_description, maxlen=maxlen)
# x_test_r = sequence.pad_sequences(x_train_r, maxlen=4)

print('Build model...')

h_description_input = Input(shape=(maxlen,))
t_description_input = Input(shape=(maxlen,))
# r_description_input = Input(shape=(4,))

h_description_embedding = Embedding(embedding_matrix.shape[0],
									embedding_dims,
									weights=[embedding_matrix],
									input_length=maxlen,
									trainable=False)(h_description_input)
t_description_embedding = Embedding(embedding_matrix.shape[0],
									embedding_dims,
									weights=[embedding_matrix],
									input_length=maxlen,
									trainable=False)(t_description_input)
# r_description_embedding = Embedding(embedding_matrix.shape[0],
# 									embedding_dims,
# 									weights=[embedding_matrix],
# 									input_length=4,
# 									trainable=False)(r_description_input)

h_description_dropout = Dropout(0.2)(h_description_embedding)
h_description_conv1 = Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1)(h_description_dropout)
h_description_maxpool1 = GlobalMaxPooling1D()(h_description_conv1)

t_description_dropout = Dropout(0.2)(t_description_embedding)
t_description_conv1 = Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1)(t_description_dropout)
t_description_maxpool1 = GlobalMaxPooling1D()(t_description_conv1)



# r_description_dropout = Dropout(0.2)(h_description_embedding)


# h_r = Dense(hidden_dims)(maxpool1)
# dropout2 = Dropout(0.2, Activation('relu'))(dense1)
cos_sim = Dot(normalize = True, axes = 1)([h_description_maxpool1, t_description_maxpool1])
output = Activation('sigmoid')(cos_sim)



model = Model(inputs = [h_description_input, t_description_input], outputs = output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([x_train_h_description, x_train_t_description], y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('cnn.h5')