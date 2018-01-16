# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import json
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm 
import time
import os
# from sklearn.decomposition import PCA
data_dir = '../../data/baike/'
def TransE():
	limit = 10
	start = 1
	x = json.load(open("./summary/candidate_len_list.json", "r"))[start:limit]
	acc_limit = json.load(open("./summary/acc_list.json", "r"))[start:limit]
	acc_base = json.load(open("./summary/acc_list_baseline.json", "r"))[start:limit]
	acc_neg1 = json.load(open("./summary/acc_list_neg1.json", "r"))[start:limit]
	acc_neg1_l2 = json.load(open("./summary/acc_list_neg1_l2.json", "r"))[start:limit]
	acc_neg1_l2_na_15 = json.load(open("./summary/acc_list_neg1_l2_na.json", "r"))[start:limit]
	acc_neg1_l2_valid = json.load(open("./summary/acc_list_neg1_l2_valid.json", "r"))[start:limit]
	acc_neg1_l2_increment = json.load(open("./summary/acc_list_neg1_l2_increment.json", "r"))[start:limit]
	acc_neg2 = json.load(open("./summary/acc_list_neg2.json", "r"))[start:limit]
	acc_neg3 = json.load(open("./summary/acc_list_neg3.json", "r"))[start:limit]
	acc_neg10 = json.load(open("./summary/acc_list_neg10.json", "r"))[start:limit]
	acc_cnn = json.load(open("./summary/acc_list_cnn.json", "r"))[start:limit]
	acc_cnn_r = json.load(open("./summary/acc_list_cnn_r.json", "r"))[start:limit]


	plt.figure()
	line_acc_limit = plt.plot(x,acc_limit, label = 'limit acc')
	line_acc_base = plt.plot(x,acc_base,label = 'TransE baseline')
	line_acc_neg1 = plt.plot(x,acc_neg1,label = 'TransE random 1 negtive')
	line_acc_neg1_l2 = plt.plot(x,acc_neg1_l2,label = 'TransE random 1 negtive l2')
	line_acc_neg1_l2_na = plt.plot(x,acc_neg1_l2_na_15,label = 'TransE random 1 negtive l2 na')
	line_acc_neg1_l2_valid = plt.plot(x,acc_neg1_l2_valid,label = 'TransE random 1 negtive l2 valid')
	line_acc_neg1_l2_increment = plt.plot(x,acc_neg1_l2_increment,label = 'TransE random 1 negtive l2 increment')
	line_acc_neg2 = plt.plot(x,acc_neg2,label = 'TransE random 2 negtive')
	line_acc_neg3 = plt.plot(x,acc_neg3,label = 'TransE random 3 negtive')
	line_acc_neg10 = plt.plot(x,acc_neg10,label = 'TransE random 10 negtive')
	line_acc_cnn = plt.plot(x,acc_cnn,label = 'cnn only description')
	line_acc_cnn_r = plt.plot(x,acc_cnn_r,label = 'cnn description and relation')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
	plt.xlabel("candidates")
	plt.ylabel("acc")
	plt.show()  

def na():
	na_dis = np.load("./summary/na_dis.npy")[:3000]
	hit_dis = np.load("./summary/hit_dis.npy")[:3000]
	plt.figure()
	plt.plot(np.abs(hit_dis), label = 'hit')
	# plt.plot(np.abs(na_dis), label = 'na')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
	plt.show()  



def get_candidate_len(candidate_list):
	len = 0 
	for c in  candidate_list:
		if c != "0":
			len += 1
	return  len 
def pca():

	ent = np.load("./summary/entity_struct_150.npy")
	rel = np.load("./summary/relation_struct_150.npy")
	pca = PCA(n_components=2)
	pca.fit(ent)
	print("start transform")
	ent = pca.transform(ent)

	pca.fit(rel)
	print("start transform")
	rel = pca.transform(rel)

	ent = np.save("./summary/entity_struct_2.npy", ent)
	rel = np.save("./summary/relation_struct_2.npy", rel)
	# plt.scatter(ent[:, 0], ent[:, 1],marker='o')
	# plt.show()
def gen_error():
	label = np.load("./summary/label.npy")
	hit = np.load("./summary/hit.npy")
	# ent = np.load("./summary/entity_struct_2.npy")
	# rel = np.load("./summary/entity_struct_2.npy")
	with open(os.path.join(data_dir, "test_with_candidate.json"), "r") as f:
		tests = json.load(f)
	id2entity = {}
	with open(os.path.join(data_dir, "entity2id.txt"), "r") as f:
		for l in f:
			entity, id = l.split()
			id2entity[int(id)] = entity
	test_len = len(tests)
	with open("error.json", "w") as f:
		for i in range(test_len):
			if label[i] == 0:
				test = tests[i]
				h, r, t, c = test
				c_len = get_candidate_len(c)
				if (c_len < 2) or (t not in c):
					continue
				c_list = []
				for candidate in c:
					if candidate != "0":
						c_list.append(candidate)
				hit_ent = id2entity[hit[i]]
				f.write(json.dumps((h,r,t,hit_ent,c_list)) + "\n")
def vis_error():
	ent = np.load("./summary/entity_struct_2.npy")
	rel = np.load("./summary/entity_struct_2.npy")
	entity2id = {}
	with open(os.path.join(data_dir, "entity2id.txt"), "r") as f:
		for l in f:
			entity, id = l.split()
			entity2id[entity] = int(id)
	relation2id = {}
	with open(os.path.join(data_dir, "relation2id.txt"), "r") as f:
		for l in f:
			relation, id = l.split()
			relation2id[relation] = int(id)
	with open("error.json", "r") as f:
		for l in f:
			h,r,t,hit_ent,c_list = json.loads(l)
			h_vec = ent[entity2id[h]]
			t_vec = ent[entity2id[t]]
			r_vec = rel[relation2id[r]]
			pre_t = h_vec + r_vec
			plt.scatter(h_vec[0],h_vec[1],marker='o')
			plt.scatter(t_vec[0],t_vec[1],marker='v')
			plt.scatter(pre_t[0],pre_t[1],marker='p')
			for c in c_list:
				plt.scatter(ent[entity2id[c]][0], ent[entity2id[c]][1],marker='^')
			plt.show()
def ana_error():
	relation = []
	relation_error = []
	head_error = []
	tail_error = []
	id2info = json.load(open("id2info.json", "r"))
	with open(os.path.join(data_dir, "test.txt"), "r") as f:
		for l in f:
			h,r,t = l.split()[:3]
			relation.append(r)
	relation_counter = Counter(relation)
	relation_dic = {}
	for key,num in relation_counter.most_common(10000):
		relation_dic[key] = num
	with open("error.json", "r") as f:
		for i, l in enumerate(f):
			h,r,t,hit_ent,c_list = json.loads(l)
			relation_error.append(r)
			if r == "所属专辑":
				print("*" * 80)
				print(id2info[h])
				print(r)
				print(id2info[t])
				print(id2info[hit_ent])
	relation_error_counter = Counter(relation_error)
	for key,num in relation_error_counter.most_common(100):
		print("|", key,"|", num, "|",round(1.0 * num/i, 2),"|", round(1.0 * num / relation_dic[key], 2), "|")

if __name__ == "__main__":
	TransE()
	# na()
	# pca()
	# ana_error()
	# vis_error()