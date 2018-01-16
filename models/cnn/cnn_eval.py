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
import json

os.environ["CUDA_VISIBLE_DEVICES"]="7"
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
model = load_model("cnn_h_r.h5")

test_h_description = np.load("test_h_description.npy")
test_candidate_description = np.load("test_candidate_description.npy")
r = np.load("test_r.npy")
r_pad = sequence.pad_sequences(r, maxlen=4)
r_extend = []
for rel in r_pad:
	for i in range(55):
		r_extend.append(rel)
r_extend = np.array(r_extend)
print("padding ...")
test_h_description = sequence.pad_sequences(test_h_description, maxlen=maxlen)
test_candidate_description = sequence.pad_sequences(test_candidate_description, maxlen=maxlen)
np.save("test_h_description_pad.npy", test_h_description)
np.save("test_candidate_description.npy", test_candidate_description)
np.save("test_r_pad.npy",r_extend)
print("start predict ...")
predict = model.predict([test_h_description,test_candidate_description, r_extend])
print("saving ...")
np.save("predict.npy", predict)



candidates = np.load("test_candidate.npy")
t_label = np.load("test_t_label.npy")
predict = -predict.reshape((-1,55))
sort_index = np.argsort(predict)

test_len = len(predict)
candidate_len = [0.0] * 56
hit_len = [0.0] * 56

for i in range(test_len):
	cur_len = np.sum(candidates[i] != 0)
	candidate_len[cur_len] += 1
	for j in range(55):

		hit_index = sort_index[i][j]

		hit1 = candidates[i][hit_index]
		if hit1 != 0:
			break
	if hit1 == t_label[i]:
		hit_len[cur_len] += 1

acc_list = []
for i in range(56):
	if candidate_len[i] != 0:
		acc_list.append(hit_len[i]/candidate_len[i])
with open("../TransE/summary/acc_list_cnn_r.json", "w") as f:
	json.dump(acc_list, f)

