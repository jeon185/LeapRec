import numpy as np
import torch
from tqdm import tqdm


class PointWiseFeedForward(torch.nn.Module):
    """
    Class of point-wise feed forward network
    """
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class LeapRec(torch.nn.Module):
    """
    Class of LeapRec's backbone model
    """
    def __init__(self, num_users, num_items, conf, user_seq_valid, user_seq_test, item_genre_mat, decay_factor):
        super(LeapRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.conf = conf
        self.device = conf['device']
        self.user_seq_valid = user_seq_valid
        self.user_seq_test = user_seq_test
        self.item_genre_mat = item_genre_mat.to_dense().to(self.device)
        self.num_genres = self.item_genre_mat.shape[1]
        self.decay_factor = decay_factor.to(self.device)

        self.item_emb = torch.nn.Embedding(self.num_items + 1, conf['hidden_dim'], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(conf['max_len'], conf['hidden_dim']) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=conf['dropout'])

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(conf['hidden_dim'], eps=1e-8)

        for _ in range(conf['blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(conf['hidden_dim'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(conf['hidden_dim'],
                                                         conf['heads'],
                                                         conf['dropout'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(conf['hidden_dim'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(conf['hidden_dim'], conf['dropout'])
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        Infer feature vectors for the sequence
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        Compute logits from sequences in training
        """
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        seq_genre_prob = self.item_genre_mat[log_seqs]
        decay_factor = self.decay_factor.unsqueeze(0).unsqueeze(2)
        decay_factor = decay_factor.expand(seq_genre_prob.shape[0], -1, seq_genre_prob.shape[2])
        seq_genre_prob = seq_genre_prob * decay_factor
        seq_p_genre = torch.cumsum(seq_genre_prob, dim=1)
        denorm = seq_p_genre.sum(2, keepdim=True)
        seq_p_genre = seq_p_genre / denorm
        seq_p_genre = torch.nan_to_num(seq_p_genre, nan=1/self.num_genres)

        coef = 1e-5
        uniform_mat = torch.full(seq_p_genre.shape, 1/self.num_genres).to(self.device)
        log_seqs_p = (1 - coef) * seq_p_genre + coef * uniform_mat

        pos_q = self.item_genre_mat[pos_seqs]
        pos_q = (1 - self.conf['beta']) * pos_q + self.conf['beta'] * log_seqs_p
        pos_kl = torch.sum(log_seqs_p * (log_seqs_p / pos_q).log(), dim=2)

        neg_q = self.item_genre_mat[neg_seqs]
        neg_q = (1 - self.conf['beta']) * neg_q + self.conf['beta'] * log_seqs_p
        neg_kl = torch.sum(log_seqs_p * (log_seqs_p / neg_q).log(), dim=2)

        return pos_logits, neg_logits, pos_logits - pos_kl, neg_logits - neg_kl

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Compute logits from sequences in inference
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def propagate(self, test):
        """
        Get user and item features to facilitate reranking
        """
        user_feats = []
        users = range(self.num_users+1)
        batch_size = self.conf['batch_size']
        for start_idx in tqdm(range(0, len(users), batch_size)):
            end_idx = min(start_idx + batch_size, len(users))
            u_batch = users[start_idx:end_idx]
            user_seq = self.user_seq_test[u_batch]
            if len(user_seq.shape) < 2:
                user_seq = [user_seq]
            user_feat = self.log2feats(np.array(user_seq))[:, -1, :].detach().cpu()
            user_feats.append(user_feat)
        user_feature = torch.cat(user_feats, dim=0)
        item_feature = self.item_emb.weight.detach().cpu()
        return user_feature, item_feature

    def evaluate(self, propagate_result, users):
        """
        Compute the relevance scores for reranking
        """
        users_feature, items_feature = propagate_result
        users_feature = users_feature[users]
        pred = torch.mm(users_feature.to(self.device), items_feature.to(self.device).t())
        return pred
