import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


class TestDataset(Dataset):
    """
    Class of test dataset
    """
    def __init__(self, pairs, graph, trn_graph):
        self.pairs = pairs
        self.graph = graph
        self.train_mask = trn_graph
        self.trn_items = np.nonzero(trn_graph.sum(0))[1]

    def __getitem__(self, index):
        grd = torch.from_numpy(self.graph[index].toarray()).squeeze()
        mask = torch.from_numpy(self.train_mask[index].toarray()).squeeze()

        return index, grd, mask, self.trn_items

    def __len__(self):
        return self.graph.shape[0]


class Datasets():
    """
    Class of all data resources
    """
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_test = conf['batch_size']
        self.num_users = conf['num_users']
        self.num_items = conf['num_items']
        self.alpha = conf['alpha']

        trn_pairs, trn_graph = self.read_file('train.txt')
        test_pairs, test_graph = self.read_file('test.txt')

        self.trn_graph = trn_graph
        self.test_data = TestDataset(test_pairs, test_graph, trn_graph)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)

        # For item categories
        genre_data_path = os.path.join(self.path, self.name)
        item_genre_path = os.path.join(genre_data_path, 'item_genres.txt')
        item_genre_df = pd.read_csv(item_genre_path)
        genre_dict_path = os.path.join(genre_data_path, 'genres_dict.txt')
        genre_dict_df = pd.read_csv(genre_dict_path)
        genre_dict = {k: v for (k, v) in zip(genre_dict_df['genre_id'], genre_dict_df['genre_name'])}
        item_genre_mat = self.df_mat(item_genre_df, genre_dict)
        item_genre_mat = item_genre_mat / item_genre_mat.sum(1)  # normalize
        item_genre_mat = spy_sparse2torch_sparse(item_genre_mat)
        user_item_mat = self.trn_graph / self.trn_graph.sum(1)  # normalize
        user_item_mat = spy_sparse2torch_sparse(user_item_mat)
        user_genre_mat = torch.sparse.mm(user_item_mat, item_genre_mat).to_dense()
        num_genres = len(genre_dict)
        user_genre_mat[user_genre_mat.sum(1) < 1e-8] = 1 / num_genres
        uniform_mat = torch.full(user_genre_mat.shape, 1 / num_genres)
        coef = 1e-5
        user_genre_mat = (1 - coef) * user_genre_mat + coef * uniform_mat
        self.user_genre_mat = user_genre_mat
        self.item_genre_mat = item_genre_mat.to_dense()

        # For sequential calibration
        seq_user_genre_mat = torch.zeros_like(user_genre_mat)
        trn_path = os.path.join(self.path, self.name, 'train.txt')
        interactions = pd.read_csv(trn_path)
        trn_grouped = interactions.groupby('user_id')
        for uid, group in trn_grouped:
            group = group.sort_values(by=['timestamp'], ascending=True)
            items = list(group['item_id'].array)[::-1]
            for i in range(len(items)):
                seq_user_genre_mat[uid] += (self.alpha ** i) * self.item_genre_mat[items[i]]
        seq_user_genre_mat[seq_user_genre_mat.sum(1) < 1e-8] = 1 / num_genres
        seq_user_genre_mat = seq_user_genre_mat / seq_user_genre_mat.sum(1).unsqueeze(1) # normalize
        seq_user_genre_mat = (1 - coef) * seq_user_genre_mat + coef * uniform_mat
        self.seq_user_genre_mat = seq_user_genre_mat

    def read_file(self, file):
        """
        Read file
        """
        path = os.path.join(self.path, self.name, file)
        with open(path, 'r') as f:
            pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(',')), f.readlines()[1:]))
        idxs = np.array(pairs, dtype=np.int32)
        values = np.ones(len(pairs), dtype=np.float32)
        graph = sp.coo_matrix(
            (values, (idxs[:, 0], idxs[:, 1])), shape=(self.num_users+1, self.num_items+1)).tocsr()
        return pairs, graph

    def df_mat(self, item_genre, genre_dict):
        """
        Convert pandas dataframe to scipy sparse matrix
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
