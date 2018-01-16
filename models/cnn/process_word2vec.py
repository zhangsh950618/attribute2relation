#-*-coding:utf-8*-

from gensim.models import word2vec
import jieba
import re
import os
import json
import numpy as np
DIRNAME = '../../data/baike/'

def train_word2vect():
	sentences = json.load(open(os.path.join(DIRNAME, "baike_sentences.json"), "r"))
	model = word2vec.Word2Vec(sentences, size=120)
	model.save("word2vec.bin")

def gen_word_index():
	 model = word2vec.Word2Vec.load('word2vec.bin')
	 print(type(model))
	 print(model['中国'])
	 for word, vec in model.items():
	 	print(word)

def get_word_index():
	word_index = json.load(open(os.path.join(DIRNAME, 'baike_word_index.json')))
	return word_index
def gen_text2vector():
	intab = "【】 ：: [].。，,“\"、；;"
	outtab = len(intab) * " "
	trantab = str.maketrans(intab, outtab)
	word_index = get_word_index()
	h_description_vs = []
	t_description_vs = []
	labels = []
	r_vs = []
	with open(os.path.join(DIRNAME, 'train_with_description_seg.json'), "r") as f:
		for i, l in enumerate(f):

			l = json.loads(l)
			h_description_v = []
			h_description = l['h_description']
			
			for sentence in h_description:
				for word in sentence:
					try:
						wi = word_index[word]
					except:

						wi = 0
					h_description_v.append(wi)
			h_description_vs.append(h_description_v)

			t_description_v = []
			t_description = l['t_description']
			for sentence in t_description:
				for word in sentence:
					try:
						wi = word_index[word]
					except:
						wi = 0
					t_description_v.append(wi)
			t_description_vs.append(t_description_v)
			r = "".join(l['r'].translate(trantab).split())
			r_v = []
			for word in jieba.cut(r):
				try:
					wi = word_index[word]
				except:
					wi = 0
				r_v.append(wi)
			r_vs.append(r_v)
			labels.append( l['label'])
	np.save("train_h_description.npy", h_description_vs)
	np.save("train_t_description.npy", t_description_vs)
	np.save("train_r.npy", r_vs)
	np.save("train_label.npy", labels)

def gen_test2vector():
	intab = "【】 ：: [].。，,“\"、；;"
	outtab = len(intab) * " "
	trantab = str.maketrans(intab, outtab)
	word_index = get_word_index()
	h_description_vs = []
	r_vs = []
	c_description_vs = []
	candidate_vs = []
	t_labels = []
	
	with open(os.path.join(DIRNAME, 'test_with_candidate_with_description_seg.json'), "r") as f:
		for i, l in enumerate(f):
			l = json.loads(l)
			h_description_v = []
			h_description = l['h_description']
			for sentence in h_description:
				for word in sentence:
					try:
						wi = word_index[word]
					except:
						wi = 0
					h_description_v.append(wi)
			candidates = l['candidates']
			candidate_v = []
			for candidate in candidates:
				c_description_v = []
				candidate_v.append(int(candidate['c']))
				for sentence in candidate['c_description']:
					for word in sentence:
						try:
							wi = word_index[word]
						except:
							wi = 0
						c_description_v.append(wi)
				h_description_vs.append(h_description_v)
				c_description_vs.append(c_description_v)
			candidate_vs.append(candidate_v)
			t_labels.append(int(l['t']))
			r = "".join(l['r'].translate(trantab).split())
			r_v = []
			for word in jieba.cut(r):
				try:
					wi = word_index[word]
				except:
					wi = 0
				r_v.append(wi)
			r_vs.append(r_v)
	np.save("test_h_description.npy", h_description_vs)
	np.save("test_candidate_description.npy", c_description_vs)
	np.save("test_r.npy", r_vs)
	np.save("test_candidate.npy", candidate_vs)
	np.save("test_t_label.npy", t_labels)

def word_seg():
	corpus = []
	sentences = []
	jieba.enable_parallel(40)
	word_list = []
	with open(os.path.join(DIRNAME, "baike.json"), "r") as f:
		for i, l in enumerate(f):
			print(i)
			entity = json.loads(l)
			description = entity['description']
			description_seg = []
			sentence_list = re.split(u'。|！|？|\?|!|', description)
			for sentence in sentence_list:
				cur_word_list = list(jieba.cut(sentence))
				description_seg.append(cur_word_list)
				word_list += cur_word_list
				sentences.append(cur_word_list)
			entity['description'] = description_seg
			corpus.append(entity)
	with open(os.path.join(DIRNAME, "baike_seg.json"), "w") as f:
		json.dump(corpus, f)
	with open(os.path.join(DIRNAME, "baike_sentences.json"), "w") as f:
		json.dump(sentences, f)
	word_list = list(set(word_list))
	word_list.insert(0,0)
	with open(os.path.join(DIRNAME, "baike_word_list.json"), "w") as f:
		json.dump(word_list, f)
	word_index = {}
	for i, word in enumerate(word_list):
		word_index[word] = i
	with open(os.path.join(DIRNAME, "baike_word_index.json"), "w") as f:
		json.dump(word_index, f)
def word_seg4train():
	# corpus = []
	# sentences = []
	# jieba.enable_parallel(40)
	# word_list = []
	print("start loading ... ")
	entities = json.load(open(os.path.join(DIRNAME, "train_with_description.json"), "r"))
	print("end  loading .. ")
	with open(os.path.join(DIRNAME, "train_with_description_seg.json"), "w") as f:
		for i, entity in enumerate(entities):
			print(i)
			if i % 5 in [2,3,4]:
				continue
			for key in ['h_description', 't_description']:
				description = entity[key]
				description_seg = []
				sentence_list = re.split(u'。|！|？|\?|!|', description)
				for sentence in sentence_list:
					cur_word_list = list(jieba.cut(sentence))
					description_seg.append(cur_word_list)
					# word_list += cur_word_list
					# sentences.append(cur_word_list)
				entity[key] = description_seg
			f.write(json.dumps(entity) + "\n")	
	# with open(os.path.join(DIRNAME, "train_sentences.json"), "w") as f:
	# 	json.dump(sentences, f)
	# word_list = list(set(word_list))
	# word_list.insert(0,0)
	# with open(os.path.join(DIRNAME, "train_word_list.json"), "w") as f:
	# 	json.dump(word_list, f)
	# word_index = {}
	# for i, word in enumerate(word_list):
	# 	word_index[word] = i
	# with open(os.path.join(DIRNAME, "train_word_index.json"), "w") as f:
	# 	json.dump(word_index, f)
def word_seg4test():
	print("start loading ... ")
	test_samples = []
	with open(os.path.join(DIRNAME, "test_with_candidate_with_description.json"), "r") as f:
		for l in f:
			test_sample = json.loads(l)
			test_samples.append(test_sample)
	print("end  loading .. ")
	with open(os.path.join(DIRNAME, "test_with_candidate_with_description_seg.json"), "w") as f:
		for test_sample in  test_samples:
			for key in ['h_description', 't_description']:
				description = test_sample[key]
				description_seg = []
				sentence_list = re.split(u'。|！|？|\?|!|', description)
				for sentence in sentence_list:
					cur_word_list = list(jieba.cut(sentence))
					description_seg.append(cur_word_list)
				test_sample[key] = description_seg
			for i, candidate in enumerate(test_sample['candidates']):
				description = candidate['c_description']
				description_seg = []
				sentence_list = re.split(u'。|！|？|\?|!|', description)
				for sentence in sentence_list:
					cur_word_list = list(jieba.cut(sentence))
					description_seg.append(cur_word_list)
				test_sample['candidates'][i]['c_description'] = description_seg
			f.write(json.dumps(test_sample) + "\n")
def gen_embedding_matrix():
	model = word2vec.Word2Vec.load('word2vec.bin')
	word_index = json.load(open(os.path.join(DIRNAME, "baike_word_index.json"), "r"))
	embedding_matrix = np.zeros((len(word_index), 120))
	for word, i in word_index.items():
		
		try:
			embedding_vector = model[word]
		except:
			embedding_vector = None
		if  embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	print(embedding_matrix[0])
	np.save('embedding_matrix.npy', embedding_matrix)

if __name__ == "__main__":
	gen_test2vector()


