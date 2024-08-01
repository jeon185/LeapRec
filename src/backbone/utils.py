import copy
import os
import random
import sys
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def moving_avg(avg, cur_num, add_value_avg, add_num):
    """
    Compute moving average
    """
    avg = (avg * cur_num + add_value_avg * add_num) / (cur_num + add_num)
    return avg


def random_neq(l, r, s):
    """
    sample random negative item
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, num_users, num_items, batch_size, max_len, result_queue, seed):
    """
    Build batch-wise data with negative sampling
    """
    def sample():
        user = np.random.randint(1, num_users+1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, num_users+1)

        seq = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = max_len - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, num_items+1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """
    Calss of sampler
    """
    def __init__(self, user_train, num_users, num_items, batch_size=64, max_len=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(user_train, num_users, num_items, batch_size, max_len, self.result_queue, np.random.randint(2e9))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(data_name, data_path):
    """
    Construct trn/valid/test datasets
    """
    user_train = {}
    user_valid = {}
    user_test = {}

    path = os.path.join(data_path, data_name)
    trn_path = os.path.join(path, 'train.txt')
    vld_path = os.path.join(path, 'valid.txt')
    tst_path = os.path.join(path, 'test.txt')
    trn_df = pd.read_csv(trn_path)
    vld_df = pd.read_csv(vld_path)
    tst_df = pd.read_csv(tst_path)

    trn_grouped = trn_df.groupby('user_id')
    for uid, group in trn_grouped:
        group = group.sort_values(by=['timestamp'], ascending=True)
        user_train[uid] = list(group['item_id'].array)
    vld_grouped = vld_df.groupby('user_id')
    for uid, group in vld_grouped:
        group = group.sort_values(by=['timestamp'], ascending=True)
        user_valid[uid] = list(group['item_id'].array)
    tst_grouped = tst_df.groupby('user_id')
    for uid, group in tst_grouped:
        group = group.sort_values(by=['timestamp'], ascending=True)
        user_test[uid] = list(group['item_id'].array)

    item_genre_path = os.path.join(path, 'item_genres.txt')
    item_genre_df = pd.read_csv(item_genre_path)
    genre_dict_path = os.path.join(path, 'genres_dict.txt')
    genre_dict_df = pd.read_csv(genre_dict_path)
    genre_dict = {k: v for (k, v) in zip(genre_dict_df['genre_id'], genre_dict_df['genre_name'])}
    item_genre_mat = get_item_genre_mat(item_genre_df, genre_dict)
    item_genre_mat = item_genre_mat / item_genre_mat.sum(1)  # normalize
    item_genre_mat = spy_sparse2torch_sparse(item_genre_mat).to_dense()

    return [user_train, user_valid, user_test, item_genre_mat]


def spy_sparse2torch_sparse(data):
    """
    Convert scipy sparse tensor to torch sparse tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor(np.array([coo_data.row, coo_data.col]))
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def get_item_genre_mat(item_genre, genre_dict):
    """
    Construct item-genre matrix
    """
    item_ids = []
    genre_ids = []
    for _, row in item_genre.iterrows():
        item_id, genre_id = row['item_id'], str(row['genre_id'])
        try:
            genre_id = [int(e) for e in genre_id.split('|')]
        except:
            genre_id = list(range(len(genre_dict)))
        item_ids += [item_id] * len(genre_id)
        genre_ids += genre_id
    values = np.ones(len(item_ids), dtype=np.float32)
    mat = sp.coo_matrix(
        (values, (item_ids, genre_ids)), shape=(item_genre.shape[0]+1, len(genre_dict))).tocsr()
    return mat


def get_user_seq_test(conf, num_users, user_train, user_valid):
    """
    Construct testing sequential dataset
    """
    placeholder = np.zeros((num_users + 1, conf['max_len']), dtype=np.int32)
    users = range(1, num_users + 1)
    for u in users:
        seq = np.zeros([conf['max_len']], dtype=np.int32)
        idx = conf['max_len'] - 1
        if u not in user_valid or u not in user_train:
            continue
        seq[idx] = user_valid[u][0]
        idx -= 1
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        placeholder[u] = seq
    return placeholder


def get_user_seq_valid(conf, num_users, user_train):
    """
    Construct validating sequential dataset
    """
    placeholder = np.zeros((num_users + 1, conf['max_len']), dtype=np.int32)
    users = range(1, num_users + 1)
    for u in users:
        seq = np.zeros([conf['max_len']], dtype=np.int32)
        idx = conf['max_len'] - 1
        if u not in user_train:
            continue
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        placeholder[u] = seq
    return placeholder


def evaluate(model, dataset, user_seq_test, num_users, num_items):
    """
    Evaluate the model on test dataset
    """
    [user_train, user_valid, user_test, item_genre_mat] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if num_users > 10000:
        users = random.sample(range(1, num_users+1), 10000)
    else:
        users = range(1, num_users+1)
    for u in users:

        if u not in user_test or u not in user_valid or u not in user_train or len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue

        seq = user_seq_test[u]
        rated = set(user_train[u])
        rated.add(0)
        item_idx = [user_test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, num_items+1)
            while t in rated: t = np.random.randint(1, num_items+1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, user_seq_valid, num_users, num_items):
    """
    Evaluate the model on validation dataset
    """
    [user_train, user_valid, user_test, item_genre_mat] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if num_users > 10000:
        users = random.sample(range(1, num_users+1), 10000)
    else:
        users = range(1, num_users+1)
    for u in users:
        if u not in user_test or u not in user_valid or u not in user_train or len(user_train[u]) < 1 or len(user_valid[u]) < 1:
            continue
        seq = user_seq_valid[u]
        rated = set(user_train[u])
        rated.add(0)
        item_idx = [user_valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, num_items+1)
            while t in rated: t = np.random.randint(1, num_items+1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
