#-*-coding-*-

import tensorflow as tf
import numpy as np
import os
import argparse
import math
import os.path
from multiprocessing import JoinableQueue,Queue,Process
import timeit 
import json
class TransE:
    @property
    def n_entity(self):
        return self.__n_entity
    @property
    def n_relation(self):
        return self.__n_relation
    @property
    def n_train(self):
        return self.__train_triple.shape[0]
    @property
    def trainable_variables(self):
        return self.__trainable
    @property
    def hr_t(self):
        return self.__hr_t
    @property
    def tr_h(self):
        return self.__tr_h
    @property
    def ht_r(self):
        return self.__ht_r
    @property
    def train_hr_t(self):
        return self.__train_hr_t
    @property
    def train_tr_h(self):
        return self.__train_tr_h
    @property
    def train_ht_r(self):
        return self.__train_ht_r
    @property
    def left_num(self):
        return self.__left_num
    @property
    def right_num(self):
        return self.__right_num
    @property
    def ent_embedding(self):
        return self.__ent_embedding
    @property
    def rel_embedding(self):
        return self.__rel_embedding

    def raw_training_data(self,batch_size=100):
        n_triple=len(self.__train_triple)
        rand_idx=np.random.permutation(n_triple)

        start=0
        while start<n_triple:
            end=min(start+batch_size,n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start=end
    
    def testing_data(self,batch_size=100):
        n_triple=len(self.__test_triple)
        start=0
        while start < n_triple:
            end=min(start+batch_size,n_triple)
            yield self.__test_triple[start:end,:]
            start=end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end
    
    def candidate_data(self, batch_size=100):
        n_triple = len(self.__candidate_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield (self.__candidate_triple[start:end, :],self.__candidates[start:end, :])
            start = end

    def load_triple(self,path):
        with open(path,'r',encoding='utf-8') as f_triple:
            #(h,t,r)
            tripes = []
            for triple in f_triple:
                h, r, t= triple.replace("\n"," ").replace("\t"," ").split(" ")[0:3]
                # print("h: ", h)
                # print("r: ", r)
                # print("t: ", t)
                tripes.append([self.__entity_id_map[h], self.__entity_id_map[t], self.__relation_id_map[r]])
            return np.asarray(tripes,dtype=np.int32)
    def load_candidate(self,path):
        f_candidate = json.load(open(path,'r',encoding='utf-8'))
        candidates = []
        triples = [] 
        for (h,r,t,candidate) in f_candidate:
            # print(t,candidate)

            candidates.append([self.__entity_id_map[c]  for c in candidate])
            triples.append([self.__entity_id_map[h], self.__entity_id_map[t], self.__relation_id_map[r]])
        # print(candidates)
        return np.asarray(triples,dtype=np.int32), np.asarray(candidates,dtype=np.int32)


    def gen_relation_attr(self):
        
        left_entity={}
        right_entity={}
        left_num={}
        right_num={}
        hrt=self.__train_triple
        relation_data=self.__relation_id_map

        for (r,r_idx) in relation_data.items():
            left_entity[r_idx]=dict()
            right_entity[r_idx]=dict()

        for h,t,r in hrt:
            if t not in right_entity[r]:
                right_entity[r][t]=0
            if h not in left_entity[r]:
                left_entity[r][h]=0

            right_entity[r][t] = right_entity[r][t] + 1
            left_entity[r][h] = left_entity[r][h] + 1
        
        for (r,r_idx) in relation_data.items():
            left_sum1=left_sum2=0
            right_sum1=right_sum2=0
            for (entity,count) in right_entity[r_idx].items():
                right_sum1=right_sum1+1
                right_sum2=right_sum2+count
            right_num[r_idx]=right_sum2/(right_sum1 + 1)

            for (entity,count) in left_entity[r_idx].items():
                left_sum1=left_sum1+1
                left_sum2=left_sum2+count
            left_num[r_idx]=left_sum2/(left_sum1 + 1)

            #print('%.3f %.3f %.3f'%(r_idx,right_num[r_idx],left_num[r_idx]))
       
        #print(len(right_num))
        return left_num,right_num  
    
    #map<int,map<int,set<int> > >
    def gen_hr_t(self,triple_data):
        hr_t=dict()
        for h,t,r in triple_data:
            if h not in hr_t:
                hr_t[h]=dict()
            if r not in hr_t[h]:
                hr_t[h][r]=set()
            hr_t[h][r].add(t)
        
        return hr_t
    
    def gen_tr_h(self,triple_data):
        tr_h=dict()
        for h,t,r in triple_data:
            if t not in tr_h:
                tr_h[t]=dict()
            if r not in tr_h[t]:
                tr_h[t][r]=set()
            tr_h[t][r].add(h)
    
        return tr_h
    
    def gen_ht_r(self,triple_data):
        ht_r=dict()
        for h,t,r in triple_data:
            if h not in ht_r:
                ht_r[h]=dict()
            if t not in ht_r[h]:
                ht_r[h][t]=set()
            ht_r[h][t].add(r)
    
        return ht_r

    def train(self,inputs,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()
            
            pos_triple,neg_triple=inputs
            
            #正样本
            pos_h_e = tf.nn.embedding_lookup(self.__ent_embedding,pos_triple[:,0])
            pos_t_e = tf.nn.embedding_lookup(self.__ent_embedding,pos_triple[:,1])
            pos_r_e = tf.nn.embedding_lookup(self.__rel_embedding,pos_triple[:,2])
            
            #负样本
            neg_h_e = tf.nn.embedding_lookup(self.__ent_embedding,neg_triple[:,0])
            neg_t_e = tf.nn.embedding_lookup(self.__ent_embedding,neg_triple[:,1])
            neg_r_e = tf.nn.embedding_lookup(self.__rel_embedding,neg_triple[:,2])

            if self.__L1_flag:
                pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e),1,keep_dims=True)
                neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e),1,keep_dims=True)
            else:
                pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e)**2,1,keep_dims=True)
                neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e)**2,1,keep_dims=True)

            loss = tf.reduce_sum(tf.maximum(pos - neg + self.__margin, 0))         
            return loss

    #inputs:shape (?,3)
    def test(self,_inputs,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            #获取实体(h,t)的CNN向量以及关系向量
            print("Get entity's cnn embedding and relation embedding")

            inputs=tf.reshape(_inputs,[-1,3,1])

            h=tf.nn.embedding_lookup(self.__ent_embedding,inputs[:,0])
            t=tf.nn.embedding_lookup(self.__ent_embedding,inputs[:,1])
            r=tf.nn.embedding_lookup(self.__rel_embedding,inputs[:,2])
            
            hrt_res=tf.reduce_sum(-abs(h+r-self.__ent_embedding),2)
            trh_res=tf.reduce_sum(-abs(r-t+self.__ent_embedding),2)
            htr_res=tf.reduce_sum(-abs(h-t+self.__rel_embedding),2)
    
            _,tail_ids=tf.nn.top_k(hrt_res,k=self.__n_entity)
            _,head_ids=tf.nn.top_k(trh_res,k=self.__n_entity)
            _,relation_ids=tf.nn.top_k(htr_res,k=self.__n_relation)
        
        return head_ids,tail_ids,relation_ids
    

    def pre_in_candidates(self,_inputs,candidate, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()
            #获取实体(h,t)的CNN向量以及关系向量
            print("Get entity's cnn embedding and relation embedding")

            inputs=tf.reshape(_inputs,[-1,3,1])

            h=tf.nn.embedding_lookup(self.__ent_embedding,inputs[:,0])
            t=tf.nn.embedding_lookup(self.__ent_embedding,inputs[:,1])
            r=tf.nn.embedding_lookup(self.__rel_embedding,inputs[:,2])
            c=tf.nn.embedding_lookup(self.__ent_embedding,candidate)
            hrt_res=tf.reduce_sum(-abs(h+r-c),2)
            _,tail_ids=tf.nn.top_k(hrt_res,k=55)
        
        return tail_ids, hrt_res


    def normalize_embedding(self):
        
        normalize_entity_op = self.__ent_embedding.assign(tf.clip_by_norm(self.__ent_embedding, clip_norm=1, axes=1))
        normalize_relation_op = self.__rel_embedding.assign(tf.clip_by_norm(self.__rel_embedding, clip_norm=1, axes=1))

        return normalize_entity_op,normalize_relation_op
        
    def __init__(self,data_dir,train_batch,eval_batch,L1_flag,margin,dim):
        
        self.__L1_flag = L1_flag
        self.__margin = margin
        self.__train_batch = train_batch
        self.__eval_batch = eval_batch
        self.__initialized = False
        self.__trainable = list()
        self.n = dim # output embedding size

        with open(os.path.join(data_dir,'entity2id.txt'),'r',encoding='utf-8') as f:
            self.__entity_id_map={x.replace("\t"," ").replace("\n"," ").split(' ')[0]: int(x.replace("\t"," ").replace("\n"," ").split(' ')[1]) for x in f.readlines()}
            self.__id_entity_map={v: k for k,v in self.__entity_id_map.items()}
        
        self.__n_entity=len(self.__entity_id_map)

        print("ENTITY: %d" % self.__n_entity)

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__relation_id_map = {x.replace("\t"," ").replace("\n"," ").split(' ')[0]: int(x.replace("\t"," ").replace("\n"," ").split(' ')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}
        self.__n_relation = len(self.__relation_id_map)

        print("RELATION: %d" % self.__n_relation)

        self.__train_triple=self.load_triple(os.path.join(data_dir,'valid.txt'))
        print("TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__valid_triple=self.load_triple(os.path.join(data_dir,'train.txt'))
        print("VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__test_triple=self.load_triple(os.path.join(data_dir,'test.txt'))
        print("TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__candidate_triple, self.__candidates=self.load_candidate(os.path.join(data_dir,'test_with_candidate.json'))
        print("CANDIDATE_TRIPLES: %d" % self.__candidate_triple.shape[0])
        self.__train_hr_t=self.gen_hr_t(self.__train_triple)
        self.__train_tr_h=self.gen_tr_h(self.__train_triple)
        self.__train_ht_r=self.gen_ht_r(self.__train_triple)
        self.__test_hr_t=self.gen_hr_t(self.__test_triple)
        self.__test_tr_h=self.gen_tr_h(self.__test_triple)
        self.__test_ht_r=self.gen_ht_r(self.__test_triple)

        self.__hr_t=self.gen_hr_t(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))
        self.__tr_h=self.gen_tr_h(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))
        self.__ht_r=self.gen_ht_r(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))

        self.__left_num,self.__right_num=self.gen_relation_attr();

        bound=6.0/math.sqrt(self.n)

        with tf.device('/gpu'):
             self.__ent_embedding = tf.get_variable(name = "ent_embedding", shape = [self.__n_entity, self.n],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=344))
             self.__rel_embedding = tf.get_variable(name = "rel_embedding", shape = [self.__n_relation, self.n],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=346))

             self.__trainable.append(self.__rel_embedding)
             self.__trainable.append(self.__ent_embedding)

def train_ops(model: TransE,learning_rate=0.01,optimizer_str='adma'):
    with tf.device('/gpu'):

        pos_triple=tf.placeholder(tf.int32,[None,3])
        neg_triple=tf.placeholder(tf.int32,[None,3])

        train_loss=model.train([pos_triple,neg_triple])
 
        if optimizer_str == 'gradient':
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'adam':
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'rms':
            optimizer=tf.train.RMSPropOptmizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Don's support %s optmizer" % optimizer_str)
        
        grads=optimizer.compute_gradients(train_loss,model.trainable_variables)

        op_train=optimizer.apply_gradients(grads)

        return pos_triple,neg_triple,train_loss,op_train
def test_ops(model: TransE):
    with tf.device('/gpu'):
        test_input=tf.placeholder(tf.int32,[None,3])
        head_ids,tail_ids,relation_ids=model.test(test_input) 
        return test_input,head_ids,tail_ids,relation_ids

def pre_in_candidates_ops(model: TransE):
    with tf.device('/gpu'):
        candidate_test_input=tf.placeholder(tf.int32,[None,3])
        candidate_input=tf.placeholder(tf.int32,[None,55])
        candidate_tail_ids, candidate_hrt_res=model.pre_in_candidates(candidate_test_input, candidate_input) 
        return candidate_test_input, candidate_input, candidate_tail_ids, candidate_hrt_res

def normalize_ops(model: TransE):
    with tf.device('/gpu'):
        return model.normalize_embedding() 

def data_generator_func(in_queue: JoinableQueue,out_queue: Queue,right_num,left_num,tr_h,hr_t,ht_r,n_entity,n_relation):
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        pos_triple_batch = []
        neg_triple_batch = []
        
        neg_rel_triple_batch=dat.copy()
        htr=dat.copy()


        for i in range(1):
            tmp_pos_triple_batch= list(dat.copy())
            tmp_neg_entity_triple_batch = list(dat.copy())
            #construct negative-triple
            for idx in range(htr.shape[0]):
                h=htr[idx,0]
                t=htr[idx,1]
                r=htr[idx,2]
                tmp_t=np.random.randint(0,n_entity-1)
                while tmp_t in hr_t[h][r]:
                    tmp_t=np.random.randint(0,n_entity-1)
                tmp_neg_entity_triple_batch[idx][1]=tmp_t
            pos_triple_batch += tmp_pos_triple_batch
            neg_triple_batch += tmp_neg_entity_triple_batch
        out_queue.put((np.asarray(pos_triple_batch),np.asarray(neg_triple_batch)))

def candidate_evaluation(testing_data,candidates, tail_pred,tr_h,hr_t,ht_r):
    

    if len(testing_data)!=len(tail_pred):
        tail_pred = tail_pred[0]
    assert len(testing_data)==len(tail_pred)
    mean_rank_t=list()

    testing_len=len(testing_data)    
    
    for i in range(testing_len):
        if candidates[i][1] == 0:
            continue
        h=testing_data[i,0]
        t=testing_data[i,1]
        r=testing_data[i,2]
        mr = 0
        flag = False
        for val in tail_pred[i]:
            if candidates[i][val] == t:
                mean_rank_t.append(mr)
                flag = True
                break
            if candidates[i][val] != 0:
                mr+=1
        if flag is False:
            mean_rank_t.append(55)
    return mean_rank_t
def candidate_evaluation_distance(testing_data,candidates, tail_pred, hrt_res):
    
    # tail_pred = tail_pred[0]
    print(np.asarray(testing_data).shape)
    print(np.asarray(candidates).shape)
    print(np.asarray(tail_pred).shape)
    print(np.asarray(hrt_res).shape)
    assert len(testing_data)==len(tail_pred)

    na_dis = list()
    hit_dis = list()

    testing_len=len(testing_data)
    candidate_len = [0 for i in  range(56)]
    hit = [0.0 for i in range(56)]
    
    for i in range(testing_len):
        h=testing_data[i,0]
        t=testing_data[i,1]
        r=testing_data[i,2]
        c_len = np.sum(np.asarray(candidates[i]) != 0)
        candidate_len[c_len] += 1
        for j in range(55):
            hit_index = tail_pred[i][j]
            hit1 = candidates[i][hit_index]
            dis = hrt_res[i][hit_index]
            if hit1 != 0:
                break
        if t in candidates[i]:
            hit_dis.append(dis)
        else:
            na_dis.append(dis)
        if dis < -18: # NA
            if t not in candidates[i]:
                hit[c_len] += 1
        else:
            if hit1 == t:
                hit[c_len] += 1
    # for i in range(55):
    #     print("|",i,"|", candidate_len[i], "|", hit[i], "|",hit[i]/candidate_len[i],"|")
    return np.asarray(candidate_len), np.asarray(hit), hit_dis, na_dis

def candidate_evaluation4diflen(testing_data,candidates, tail_pred):
    
    tail_pred = tail_pred[0]
    assert len(testing_data)==len(tail_pred)
    mean_rank_t=list()

    testing_len=len(testing_data)
    candidate_len = [0 for i in  range(56)]
    hit = [0.0 for i in range(56)]
    
    for i in range(testing_len):
        h=testing_data[i,0]
        t=testing_data[i,1]
        r=testing_data[i,2]
        c_len = np.sum(np.asarray(candidates[i]) != 0)
        candidate_len[c_len] += 1
        for j in range(55):
            hit1 = candidates[i][tail_pred[i][j]]
            if hit1 != 0:
                break
        if hit1 == t:
            hit[c_len] += 1
    # for i in range(55):
    #     print("|",i,"|", candidate_len[i], "|", hit[i], "|",hit[i]/candidate_len[i],"|")
    return np.asarray(candidate_len), np.asarray(hit)


def test_evaluation(testing_data,head_pred,tail_pred,relation_pred,tr_h,hr_t,ht_r):
    assert len(testing_data)==len(head_pred)
    assert len(testing_data)==len(tail_pred)
    assert len(testing_data)==len(relation_pred)

    mean_rank_h=list()
    mean_rank_t=list()
    mean_rank_r=list()

    filtered_mean_rank_h=list()
    filtered_mean_rank_t=list()
    filtered_mean_rank_r=list()

    testing_len=len(testing_data)    
    
    for i in range(testing_len):
        h=testing_data[i,0]
        t=testing_data[i,1]
        r=testing_data[i,2]
    
    # mean rank - predict head entity
        mr=0
        for val in head_pred[i]:
            if val == h:
                mean_rank_h.append(mr)
                break
            mr+=1

        # mean rank - predict tail entity
        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
                break
            mr+=1

        # mean rank - predict relation 
        mr = 0
        for val in relation_pred[i]:
            if val == r:
                mean_rank_r.append(mr)
                break
            mr+=1

        #filtered mean rank - predict head entity
        fmr=0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

        #filtered mean rank - predict tail entity
        fmr=0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1
        
        #filtered mean rank - predict relation
        fmr=0
        for val in relation_pred[i]:
            if val == r:
                filtered_mean_rank_r.append(fmr)
                break
            if h in ht_r and t in ht_r[h] and val in ht_r[h][t]:
                continue
            else:
                fmr += 1

    return (mean_rank_h,filtered_mean_rank_h),(mean_rank_t,filtered_mean_rank_t),(mean_rank_r,filtered_mean_rank_r) 

def worker_func(in_queue: JoinableQueue, out_queue: Queue,tr_h,hr_t,ht_r):
    while True:
        dat=in_queue.get()
        if dat is None:
            in_queue.task_done()
            continue
        testing_data,candidates,candidate_tail_pred = dat
        out_queue.put(candidate_evaluation(testing_data,candidates,candidate_tail_pred,tr_h,hr_t,ht_r))
        in_queue.task_done()

def main(_):
    parser = argparse.ArgumentParser(description='TransE.')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.005)
    parser.add_argument('--L1_flag', dest='L1_flag', type=int, help="norm method", default=0)
    parser.add_argument('--margin', dest='margin', type=int, help="margin", default=1)
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='../../data/baike/')
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=150)
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file:xxx.meta", default=None)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=500)
    parser.add_argument("--train_batch", dest="train_batch", type=int, help="Training batch size", default=10240)
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=40960)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='gradient')
    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=10)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='./log/')
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=50)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=5)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",default='./TransE_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",default=0.5)
    parser.add_argument("--pad", dest='candidate_dim', type=int, help="dimension of the candidate",default=55)
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='neg 1 valid')
    args=parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    model=TransE(data_dir=args.data_dir,train_batch=args.train_batch,eval_batch=args.eval_batch,L1_flag=args.L1_flag,margin=args.margin, dim = args.dim)
    pos_triple,neg_triple,train_loss,train_op = train_ops(model,learning_rate=args.lr,optimizer_str=args.optimizer)
    test_input,test_head,test_tail,test_relation=test_ops(model)
    normalize_entity_op,normalize_relation_op=normalize_ops(model)
    candidate_test_input,candidate_input,candidate_tail_ids, candidate_hrt_res = pre_in_candidates_ops(model)

    init=tf.global_variables_initializer()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    saver = tf.train.Saver()

    # run tensorflow
    with tf.Session(config=config) as sess:
            sess.run(init)
            iter_offset = 0
            total_inst=model.n_train
            raw_training_data_queue = Queue()
            training_data_queue = Queue()
            evaluation_queue=JoinableQueue()
            result_queue=Queue()
            data_generators = list()
            for i in range(args.n_generator):
                data_generators.append(Process(target=data_generator_func,args=(
                                raw_training_data_queue,training_data_queue,model.right_num,model.left_num,model.train_tr_h,model.train_hr_t,model.train_ht_r,
                                model.n_entity,model.n_relation)))
                data_generators[-1].start()
            for i in range(args.n_worker):
                worker=Process(target=worker_func,args=(evaluation_queue,result_queue,
                               model.tr_h,model.hr_t,model.ht_r))
                worker.start()
                print("work %d start!"% i)
            for n_iter in range(iter_offset,args.max_iter):
                start_time = timeit.default_timer()
                accu_loss = 0.0
                ninst = 0
                nbatches_count = 0
                for dat in model.raw_training_data(batch_size=args.train_batch):
                    raw_training_data_queue.put(dat)
                    nbatches_count += 1
                while nbatches_count > 0:

                    nbatches_count -= 1

                    pos_triple_batch,neg_triple_batch= training_data_queue.get()

                    sess.run([normalize_entity_op,normalize_relation_op])

                    loss, _= sess.run([train_loss,train_op], feed_dict={pos_triple:pos_triple_batch,
                                                                        neg_triple:neg_triple_batch})
                    accu_loss += loss
                    ninst += pos_triple_batch.shape[0]

                    if ninst % (5000) is not None:
                        print(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            loss / (pos_triple_batch.shape[0])),end='\r')
                print("iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

                if n_iter != 0 and (n_iter % args.save_per == 0 or n_iter == args.max_iter - 1):
                    save_path=saver.save(sess,os.path.join(
                        args.save_dir,"TransE_"+str(args.prefix)+"_"+str(n_iter)+".ckpt"))
                    print("TransE Model saved at %s" % save_path)

                if n_iter!=0 and (n_iter % args.eval_per == 0 or n_iter == args.max_iter - 1):
                    test_start_time=timeit.default_timer()
                    accu_mean_rank_t = list()
                    hit_dis = []
                    na_dis = []
                    evaluation_count = 0
                    candidate_len = [0 for i in  range(56)]
                    hit = [0.0 for i in range(56)]
                    for (testing_data,candidates) in model.candidate_data(batch_size=args.eval_batch):
                        candidate_tail_pred, candidate_tail_res =sess.run([candidate_tail_ids, candidate_hrt_res],{candidate_test_input: testing_data,candidate_input:candidates})
                        temp_candidate_len , temp_hit, temp_hit_dis, temp_na_dis = candidate_evaluation_distance(testing_data,candidates, candidate_tail_pred, candidate_tail_res)
                        candidate_len += temp_candidate_len
                        hit += temp_hit
                        hit_dis += temp_hit_dis
                        na_dis += temp_na_dis
                        evaluation_queue.put((testing_data,candidates,candidate_tail_pred))
                        evaluation_count += 1
                    print(hit_dis)
                    np.save("./summary/hit_dis.npy", hit_dis)
                    np.save("./summary/na_dis.npy", na_dis)
                    candidate_len_list = []
                    acc_list = []

                    for i in range(56):
                        if candidate_len[i] != 0:
                            print("|",i,"|", candidate_len[i],"|", round(candidate_len[i]/408867.0,3), "|", hit[i], "|",round(hit[i]/candidate_len[i],3),"|")
                            candidate_len_list.append(i)
                            acc_list.append(round(hit[i]/candidate_len[i],3))
                    print("hit_dis: ", np.mean(np.asarray(hit_dis)))
                    print("na_dis: ", np.mean(np.asarray(na_dis)))
                    with open("./summary/candidate_len_list.json", "w") as f:
                        json.dump(candidate_len_list, f)
                    with open("./summary/acc_list_neg1_l2_valid.json", "w") as f:
                        json.dump(acc_list, f)

                    for i in range(args.n_worker):
                        evaluation_queue.put(None)
                    print("waiting for worker finishes their work")
                    evaluation_queue.join()
                    print("all worker stopped.")

                    while evaluation_count > 0:
                        evaluation_count -= 1
                        mrt = result_queue.get()
                        accu_mean_rank_t += mrt
                    print('cost time:[%.3f sec]'%(timeit.default_timer()-test_start_time))
                    print("margin: ", args.margin, "dim : ", args.dim, "batch_size", args.train_batch)
                    print(
                        "INITIALIZATION [TAIL PREDICTION] MEAN RANK: %.1f  HIT@10 %.3f HIT@1 %.3f tail not in candidate %.3f" %
                        (np.mean(accu_mean_rank_t), 
                        np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                        np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 1),
                        np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) == 55)))
            for i in range(args.n_generator):
                data_generators[i].terminate()
            # for i in range(args.n_worker):
            #     data_generators[i].terminate()
                     

if __name__ == "__main__":
    tf.app.run()




