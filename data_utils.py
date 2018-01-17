# -*-coding:utf-8-*-
from pymongo import MongoClient
from pymongo.cursor import Cursor
import json
from collections import Counter
import random
from bson.objectid import ObjectId
import re
import ujson as json
import os
import networkx as nx 
from collections import Counter
import itertools  
import numpy as np 
DIRNAME = './data/baike/'
def get_collection():
	client = MongoClient()
	client = MongoClient('mongodb://localhost:27017/')
	db = client['baike']
	collection = db['entries']
	return collection

def attribute2relation():
	id2name = "../data/ID2Name.json"
	normalized_relation = "../data/normalized_relation_list.json"
	with open(normalized_relation, "r") as f:
		normalized_relation_list = json.load(f)
	normalized_relation_set = set(normalized_relation_list)
	name2id = dict()
	with open(id2name,"r") as f:
		for line in f:
			d = json.loads(line)
			name = d['name']
			id = d['_id']['$oid']
			normalized_name = name.split('（')[0].strip()
			try:
				_id = name2id[normalized_name]
				name2id[normalized_name] = -1
			except:
				name2id[normalized_name] = id
	collection = get_collection()
	intab = "【】 ：: [].。，,“\"、；;"
	outtab = len(intab) * " "
	trantab = str.maketrans(intab, outtab)
	value_counter = 0
	mutivalue_counter = 0
	nonevalue_counter = 0
	selfvalue_counter = 0
	for i, entity in enumerate(collection.find()):
		attribute2relations = []
		print("now is processing : ", i)
		print(entity['name'], entity['_id'])
		print("value_counter:", value_counter, "mutivalue_counter:", mutivalue_counter, "nonevalue_counter:", nonevalue_counter, "selfvalue:", selfvalue_counter)
		infoboxs = entity['infobox']
		for infobox in infoboxs:
			relation = "".join(infobox['name'].translate(trantab).split())
			values = re.split(";|,|；|，|、|\"|《|》|”",infobox['value'])
			if relation not in normalized_relation_set:
				continue
			for value in values:
				value_counter += 1
				value = value.strip()
				print(value)
				t_id = name2id.get(value)
				if t_id is not None:
					if t_id == -1:
						mutivalue_counter += 1
						print("multi")
					elif t_id == str(entity['_id']):
						selfvalue_counter += 1
						print("self")
					else:
						attribute2relation = dict()
						attribute2relation['name'] = relation
						attribute2relation['value'] = value
						attribute2relation['_id'] = t_id
						attribute2relations.append(attribute2relation)
						print("ok")
				else:
					nonevalue_counter += 1
		collection.update({'_id':entity['_id']}, {'$set': {"attribute2relation": attribute2relations}})


def export_from_mongodb():
	collection = get_collection()
	for i, entity in enumerate(collection.find()):
		if i < 2:
			print(entity)
		else:
			break



def get_normalized_relation_list():
	normalized_relation_list = []
	with open(os.path.join(DIRNAME, "normalized_relation.txt"), "r") as f:
		for line in f:
			normalized_relation_list.append(line.split()[0])
	return normalized_relation_list



def relation_count():
	intab = "【】 ：: [].。，,“\""
	outtab = len(intab) * " "
	trantab = str.maketrans(intab, outtab)
	normalized_relation_list = get_normalized_relation_list()
	id2name = json.load(open(os.path.join(DIRNAME, "id2name.json"), "r"))
	with open(os.path.join(DIRNAME, "baike.json"), "r") as f:
		with open(os.path.join(DIRNAME,"triple.txt"), "w") as w:
			for i, l in enumerate(f):
				entity = json.loads(l)
				name = entity['name']
				relations = entity['relations']
				item_id = entity['item_id']
				for relation in relations:
					if relation['source'] == 'INFOBOX':
						predicate = "".join(relation['predicate'].translate(trantab).split())
						if predicate in normalized_relation_list and id2name.get(str(relation['object_id'])) is not None:
							line = str(item_id) + " " + predicate + " " +  str(relation['object_id']) + " " + str(relation['object']) + "\n"
							w.write(line)


def build_graph_from_triple():
	G = nx.Graph()
	with open(os.path.join(DIRNAME,"triple.txt"),"r") as f:
		for line in f:
			h,r,t = line.split()[:3]
			G.add_edge(h,t)

	return G
    

def largest_connected_component():
	G = build_graph_from_triple()
	connected_components_list = sorted(nx.connected_components(G), key=len,reverse=True)
	largest_cc = max(nx.connected_components(G), key = len)
	with open(os.path.join(DIRNAME, "largest_connected_component.txt"), "w") as w:
		with open(os.path.join(DIRNAME, "triple.txt"), "r") as f:
			for line in f:
				h, r, t, mention = line.split()[:4]
				if h in largest_cc or t in largest_cc:
					s = h + " " + r + " " + t + " " + mention + "\n"
					w.write(s)

def divide_data_set():
	normalized_relation_list = get_normalized_relation_list()
	relation = {}
	entity = {}
	train = []
	valid = []
	test = []
	with open(os.path.join(DIRNAME, "largest_connected_component.txt"), "r") as f:
		f_list = []
		
		for line in f:
			h, r, t = line.split()[:3]
			entity[h] = True
			entity[t] = True
			relation[r] = True
			f_list.append(line)
		with open(os.path.join(DIRNAME, "entity2id.txt"), "w", encoding = 'utf-8') as f:
			line = str(0) + " " + str(0) + "\n"
			f.write(line)
			for i, key in enumerate(entity.keys()):
				line = str(key) + " " + str(i + 1) + "\n"
				f.write(line)
		with open(os.path.join(DIRNAME, "relation2id.txt"), "w", encoding = 'utf-8') as f:
			for i, key in enumerate(relation.keys()):
				line = str(key) + " " + str(i) + "\n"
				f.write(line)
		random.shuffle(f_list)
		f_list = list(filter(lambda x:x.split()[0]!=x.split()[2], f_list))
		train_len = int(len(f_list) * 0.6)
		valid_len = int(len(f_list) * 0.2)
		test_len = len(f_list) - train_len - valid_len
		mark = {}
		for i, line in enumerate(f_list):
			h, r, t = line.split()[:3]
			mark[i] = False
			if entity[h] == True or entity[t] == True or relation[r] == True:
				mark[i] = True
				train.append(line)
				entity[h] = False
				entity[t] = False
				relation[r] = False
		for i, line in enumerate(f_list):
			if mark[i]:
				continue
			if len(train) < train_len:
				train.append(line)
			elif len(valid) < valid_len:
				valid.append(line)
			else:
				test.append(line)
			mark[i] = True



		with open(os.path.join(DIRNAME, "train.txt"), "w") as f:
			for line in train:
				f.write(line)
		with open(os.path.join(DIRNAME, "valid.txt"), "w") as f:
			for line in valid:
				f.write(line)
		with open(os.path.join(DIRNAME, "test.txt"), "w") as f:
			for line in test:
				f.write(line)
def gen_id2name():
	id2name = {}
	with open(os.path.join(DIRNAME, "baike.json"), "r") as f:
		for line in f:
			entity = json.loads(line)
			name = entity['name']
			item_id = entity['item_id']
			id2name[item_id] = name
		with open(os.path.join(DIRNAME, "id2name.json"), "w") as f:
			json.dump(id2name, f)
def gen_name2candidate4all():
	name2candidates = {}
	normalized_relation_list = get_normalized_relation_list()
	intab = "【】 ：: [].。，,“\"、；;()（）"
	outtab = len(intab) * " "
	trantab = str.maketrans(intab, outtab)
	with open(os.path.join(DIRNAME, "baike.json"), "r") as f:
		for i, line in enumerate(f):
			print("now is processing", i)
			entity = json.loads(line)
			name = entity['name']
			item_id = entity['item_id']
			relations = entity['relations']
			if name2candidates.get(name) is None:
				name2candidates[name] = [item_id]
			else:
				name2candidates[name].append(item_id)
			for relation in relations:
					if relation['source'] == 'INFOBOX':
						predicate = "".join(relation['predicate'].translate(trantab).split())
						if predicate in ["简称","别名","别称"]:
							objects = str(relation['object'])
							for name in objects.translate(trantab).split():
								if name2candidates.get(name) is None:
									name2candidates[name] = [item_id]
								else:
									name2candidates[name].append(item_id)
	with open(os.path.join(DIRNAME, "name2candidates.json"), "w") as f:
		json.dump(name2candidates, f)
def gen_mention2candidate4all():
	mention2candidates = {}
	char2candidates = json.load(open(os.path.join(DIRNAME, "char2candidates.json"), "r"))
	with open(os.path.join(DIRNAME,"triple.txt"),"r") as f:
		for i, line in enumerate(f):
			print("now is processing ", i)
			h,r,t, mention = line.split()[:4]
			for mention_combinations in itertools.combinations(mention,int((len(mention) + 1) / 2)):
				print("mention :",mention, "mention_combinations :", mention_combinations)
				for i, mention_char in enumerate(mention_combinations):
					if i == 0:
						candidate_set = set(char2candidates[mention_char])
					else:
						candidate_set = candidate_set & set(char2candidates[mention_char])
			mention2candidates[mention] = list(candidate_set)
	with open(os.path.join(DIRNAME, "mention2candidates.json"), "w") as f:
		json.dump(mention2candidates, f)
def gen_candidate4all_from_mention_with_fullname_nickname():
	mention2candidates = {}
	char2candidates = json.load(open(os.path.join(DIRNAME, "char2candidates.json"), "r"))
	with open(os.path.join(DIRNAME,"triple.txt"),"r") as f:
		for i, line in enumerate(f):
			print("now is processing ", i)
			h,r,t, mention = line.split()[:4]
			for mention_combinations in itertools.combinations(mention,int((len(mention) + 1) / 2)):
				print("mention :",mention, "mention_combinations :", mention_combinations)
				for i, mention_char in enumerate(mention_combinations):
					if i == 0:
						candidate_set = set(char2candidates[mention_char])
					else:
						candidate_set = candidate_set & set(char2candidates[mention_char])
			mention2candidates[mention] = list(candidate_set)
	with open(os.path.join(DIRNAME, "mention2candidates.json"), "w") as f:
		json.dump(mention2candidates, f)

def gen_char2candidate4all():
	name2candidates = {}
	with open(os.path.join(DIRNAME, "baike.json"), "r") as f:
		for i, line in enumerate(f):
			entity = json.loads(line)
			name = entity['name']
			item_id = entity['item_id']
			for char in name:
				if name2candidates.get(char) is None:
					name2candidates[char] = [item_id]
				else:
					name2candidates[char].append(item_id)
	with open(os.path.join(DIRNAME, "char2candidates.json"), "w") as f:
		json.dump(name2candidates, f)

def gen_candidate4test_fb():
	dir_name = './data/FB15k/'
	candidates = []
	entity_set = set()
	with open(os.path.join(dir_name,"train.txt"), "r") as f_train:
		for i, line in enumerate(f_train):
			h,t,r = line.split()[:3]
			entity_set.add(h)
	entity_list = list(entity_set)
	with open(os.path.join(dir_name,"test.txt"), "r") as f_test:
		for i, line in enumerate(f_test):
			h,t,r = line.split()[:3]
			c = random.sample(entity_list, 100)
			c.append(t)
			candidates.append((h,r,t,c))
	with open(os.path.join(dir_name,"candidate.json"), "w") as f_candidate:
		json.dump(candidates, f_candidate)

def gen_candidate4data(filename):
	name2candidates = json.load(open(os.path.join(DIRNAME, "name2candidates.json"), "r"))
	id2name = json.load(open(os.path.join(DIRNAME, "id2name.json"), "r"))
	entity_set = set()
	with open(os.path.join(DIRNAME,"entity2id.txt"), "r") as f:
		for line in f:
			entity = line.split()[0]
			entity_set.add(entity)
	candidates = []
	with open(os.path.join(DIRNAME,filename + '.txt'), "r") as f_test:
		for i, line in enumerate(f_test):
			h,r,t,mention = line.split()[:4]
			candidate_list = name2candidates.get(mention)
			if candidate_list is None:
				candidate_list = []
			c = list(entity_set & set(candidate_list))
			c = c[:55]
			while len(c) < 55:
				c.append('0')
			# c = name2candidates[id2name[t]]
			candidates.append((h,r,t,c))
	with open(os.path.join(DIRNAME,filename+"_with_candidate.json"), "w") as f_candidate:
		json.dump(candidates, f_candidate)
def eval_candidate():
	candidates = json.load(open(os.path.join(DIRNAME,"train_with_candidate.json"), "r"))
	test_len = len(candidates)
	print("tot test len : ", test_len)
	c_list = []
	recall = 0.0
	tail_in_candidates = [0] * 56
	candidate_len = [0] * 56
	for (h,r,t,c) in candidates:
		l = 0
		for cc in c:
			if cc == '0':
				break
			l += 1
		candidate_len[l] += 1
		if t in c:
			tail_in_candidates[l] += 1
	acc_list = []
	for i in range(56):
		if candidate_len[i] != 0:
			print("|", i, "|", candidate_len[i],"|",round(candidate_len[i] / test_len, 3),"|",  tail_in_candidates[i],"|", round(tail_in_candidates[i] / candidate_len[i], 3),"|")
			acc_list.append(round(tail_in_candidates[i] / candidate_len[i], 3))
	with open("./models/TransE/summary/acc_list.json", "w") as f:
		json.dump(acc_list, f)
	print("recall : ", 1.0 * np.sum(np.array(tail_in_candidates))/ np.sum(np.array(candidate_len)))


def get_id2description():
	id2description = {}
	print("start loading ...")
	# id2description = json.load(open(os.path.join(DIRNAME, 'baike_seg.json'), "r"))
	with open(os.path.join(DIRNAME, 'baike.json'), "r") as f:
		for i, l in enumerate(f):
			print("loading ...", i)
			entity = json.loads(l)
			id2description[entity['item_id']] = entity['description']
	print("end loading")
	return id2description
def gen_triple_with_description(filename):
	id2description = get_id2description()
	triple_with_description = []
	triple = {}
	entities_list = []
	with open(os.path.join(DIRNAME, filename + '.txt'), "r") as f:
		for line in f:
			h, r, t = line.split()[:3]
			entities_list.append(h)
			entities_list.append(t)
			if triple.get(h) is None:
				triple[h] = {}
			if triple[h].get(r) is None:
				triple[h][r] = {}
			if triple[h][r].get(t) is None:
				triple[h][r][t] = True
		entities_list = list(set(entities_list))
	with open(os.path.join(DIRNAME, filename + '.txt'), "r") as f:
		for i, line in enumerate(f):
			print(i)
			h, r, t = line.split()[:3]
			d = {}
			d['h'] = h
			d['r'] = r
			d['t'] = t
			d['h_description'] = id2description[h]
			d['t_description'] = id2description[t]
			d['label'] = 1
			triple_with_description.append(d)
			for i in range(4):
				tem_t = random.choice(entities_list)
				while triple[h][r].get(tem_t) is True:
					tem_t = random.choice(entities_list)
				d = {}
				d['h'] = h
				d['r'] = r
				d['t'] = tem_t
				d['h_description'] = id2description[h]
				d['t_description'] = id2description[tem_t]
				d['label'] = 0
				triple_with_description.append(d)
	with open(os.path.join(DIRNAME, filename + '_with_description.json'), "w") as f:
		json.dump(triple_with_description,f)

def gen_candidate_with_description(filename):
	# id2description = {}
	id2description = get_id2description()
	triple_with_description = []
	triple = {}
	entities_list = []
	tests = json.load(open(os.path.join(DIRNAME, filename + '.json'), "r"))
	with open(os.path.join(DIRNAME, filename + '_with_description.json'), "w") as f:
		for i, (h,r,t,m) in enumerate(tests):
			print(i)
			d = {}
			d['h'] = h
			d['r'] = r
			d['t'] = t
			d['h_description'] = id2description[h]
			d['t_description'] = id2description[t]
			candidates = []
			for mm in m:
				md = {}
				try:
					c_description = id2description[mm]
				except:
					c_description = ''
				md['c'] = mm
				md['c_description'] = c_description
				candidates.append(md)
			d['candidates'] = candidates
			f.write(json.dumps(d) + "\n")
if __name__ == '__main__':
	random.seed(1)
	# gen_id2name()
	# gen_candidate4all()
	# relation_count()
	# largest_connected_component()
	# divide_data_set()
	# gen_triple_with_description()
	# largest_connected_component()
	# divide_data_set()
	gen_candidate4data('valid')
	# gen_candidate_with_description('test_with_candidate')
	# eval_candidate()