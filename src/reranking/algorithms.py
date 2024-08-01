import torch.nn as nn
from tqdm import tqdm

from utils import *


def form_content(test_results, ks):
    """
    Format to display
    """
    content = f'   Rec@{ks[0]} |  Rec@{ks[1]} |' \
              f'  nDCG@{ks[0]} | nDCG@{ks[1]} |' \
              f'  mCal@{ks[0]} | mCal@{ks[1]} |' \
              f' smCal@{ks[0]} |smCal@{ks[1]} |\n'
    test_content = ''
    test_results_recall = test_results['recall']
    for k in ks:
        test_content += f'  {test_results_recall[k]:.4f} |'
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'  {test_results_ndcg[k]:.4f} |'
    test_results_mcal = test_results['mcal']
    for k in ks:
        test_content += f'  {test_results_mcal[k]:.4f} |'
    test_results_smcal = test_results['smcal']
    for k in ks:
        test_content += f'  {test_results_smcal[k]:.4f} |'
    content += test_content
    return content


def get_recall(grd, is_hit):
    """
    Compute recall (hit ratio in this case)
    """
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = grd.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(grd, is_hit, topk):
    """
    Compute nDCG
    """
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float).to(device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float).to(device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = grd.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


def get_mcal(col_indice, user_genre_mat, item_genre_mat, topk, beta):
    """
    Compute miscalibration
    If p is sequential distribution, sequential miscalibration is computed
    """
    row_idxs = np.arange(0, col_indice.shape[0])[:, np.newaxis]
    row_idxs = np.tile(row_idxs, (1, col_indice.shape[1]))
    row_idxs = row_idxs.flatten()
    col_idxs = col_indice.cpu().flatten()
    values = np.full(col_idxs.shape, 1/topk)
    mat = sp.coo_matrix(
        (values, (row_idxs, col_idxs)), shape=(col_indice.shape[0], item_genre_mat.shape[0])).tocsr()
    rec_distribution = torch.sparse.mm(spy_sparse2torch_sparse(mat), item_genre_mat)
    rec_distribution = (1-beta) * rec_distribution + beta * user_genre_mat
    kl_loss = nn.KLDivLoss(reduction='sum')

    nomina = kl_loss(rec_distribution.log(), user_genre_mat)
    denorm = user_genre_mat.shape[0]

    return [nomina, denorm]


class Origin(object):
    """
    Class of algorithm that considers only relevance
    """
    def __init__(self):
        super(Origin, self).__init__()

    def get_dataset(self, model, test_loader, user_genre_mat, seq_user_genre_mat, item_genre_mat, conf):
        """
        Get datasets
        """
        self.model = model
        self.test_loader = test_loader
        self.user_genre_mat = user_genre_mat
        self.seq_user_genre_mat = seq_user_genre_mat
        self.item_genre_mat = item_genre_mat
        self.conf = conf

    def evaluate(self):
        """
        Evaluate the recommendations
        """
        metrics = self.test(self.model, self.test_loader, self.user_genre_mat, self.seq_user_genre_mat, self.item_genre_mat, self.conf)
        content = form_content(metrics, self.conf['topk'])
        print(content)

    def test(self, model, dataloader, user_genre_mat, seq_user_genre_mat, item_genre_mat, conf):
        """
        Rerank for all users
        """
        tmp_metrics = {}
        for m in ['recall', 'ndcg', 'mcal', 'smcal']:
            tmp_metrics[m] = {}
            for topk in conf['topk']:
                tmp_metrics[m][topk] = [0, 0]
        device = conf['device']
        model.eval()
        rs = model.propagate(test=True)
        for users, ground_truth, train_mask, trn_items in tqdm(dataloader):
            pred = model.evaluate(rs, users)
            pred -= 1e8 * train_mask.to(device)
            rec_items = self.algorithm(pred, conf['topk'])
            tmp_metrics = self.get_metrics(tmp_metrics, ground_truth.to(device), pred,
                                           user_genre_mat[users], seq_user_genre_mat[users],
                                           item_genre_mat, rec_items, conf['topk'])

        metrics = {}
        for m, topk_res in tmp_metrics.items():
            metrics[m] = {}
            for topk, res in topk_res.items():
                metrics[m][topk] = res[0] / res[1]

        return metrics

    def algorithm(self, pred, topks):
        """
        Reranking algorithm
        """
        _, rec_items = torch.topk(pred, max(topks))
        return rec_items

    def get_metrics(self, metrics, grd, pred, user_genre_mat, seq_user_genre_mat, item_genre_mat, rec_items, topks):
        """
        Measure the results for each metrics
        """
        tmp = {'recall': {}, 'ndcg': {}, 'mcal': {}, 'smcal': {}}
        for topk in topks:
            col_indice = rec_items[:, :topk].contiguous()
            row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                     dtype=torch.long).view(-1, 1)
            is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
            tmp['recall'][topk] = get_recall(grd, is_hit)
            tmp['ndcg'][topk] = get_ndcg(grd, is_hit, topk)
            tmp['mcal'][topk] = get_mcal(col_indice, user_genre_mat, item_genre_mat, topk, beta=self.conf['beta'])
            tmp['smcal'][topk] = get_mcal(col_indice, seq_user_genre_mat, item_genre_mat, topk, beta=self.conf['beta'])

        for m, topk_res in tmp.items():
            for topk, res in topk_res.items():
                for i, x in enumerate(res):
                    metrics[m][topk][i] += x
        return metrics


class LeapRecRerank(object):
    """
    Class of LeapRec reranking algorithm
    """
    def __init__(self, balance):
        super(LeapRecRerank, self).__init__()
        self.balance = balance

    def get_dataset(self, model, test_loader, user_genre_mat, seq_user_genre_mat, item_genre_mat, conf):
        """
        Get datasets
        """
        self.model = model
        self.test_loader = test_loader
        self.user_genre_mat = user_genre_mat
        self.seq_user_genre_mat = seq_user_genre_mat
        self.item_genre_mat = item_genre_mat
        self.conf = conf

    def evaluate(self):
        """
        Evaluate the recommendations
        """
        metrics = self.test(self.model, self.test_loader, self.user_genre_mat, self.seq_user_genre_mat, self.item_genre_mat, self.conf)
        content = form_content(metrics, self.conf['topk'])
        print(content)

    def test(self, model, dataloader, user_genre_mat, seq_user_genre_mat, item_genre_mat, conf):
        """
        Rerank for all users
        """
        tmp_metrics = {}
        for m in ['recall', 'ndcg', 'mcal', 'smcal']:
            tmp_metrics[m] = {}
            for topk in conf['topk']:
                tmp_metrics[m][topk] = [0, 0]
        device = conf['device']
        model.eval()
        rs = model.propagate(test=True)
        for users, ground_truth, train_mask, trn_items in tqdm(dataloader):
            pred = model.evaluate(rs, users)
            pred -= 1e8 * train_mask.to(device)
            rec_items, acc_gains, cali_gains = self.algorithm(pred, user_genre_mat[users], seq_user_genre_mat[users], item_genre_mat, conf['topk'])
            rec_items = rec_items.to(device)
            tmp_metrics = self.get_metrics(tmp_metrics, ground_truth.to(device), pred,
                                           user_genre_mat[users], seq_user_genre_mat[users],
                                           item_genre_mat, rec_items, conf['topk'])

        metrics = {}
        for m, topk_res in tmp_metrics.items():
            metrics[m] = {}
            for topk, res in topk_res.items():
                metrics[m][topk] = res[0] / res[1]

        return metrics

    def algorithm(self, pred, user_genre_mat, seq_user_genre_mat, item_genre_mat, topks):
        """
        Reranking algorithm
        """
        rec_items = []
        acc_gains = []
        cali_gains = []
        q = None
        acc_gain = pred.cpu().detach()
        for k in range(max(topks)):
            cali_gain = self.get_cali_gain(seq_user_genre_mat, item_genre_mat, q, k, self.balance)
            balance = self.balance ** (1/(k+1))
            gain = (1 - balance) * acc_gain + balance * cali_gain

            _, col_indice = torch.topk(gain, 1)
            rec_items.append(col_indice)

            acc_values = torch.gather(acc_gain, 1, col_indice)
            acc_gains.append(acc_values)
            cali_values = torch.gather(cali_gain, 1, col_indice)
            cali_gains.append(cali_values)

            acc_gain[list(range(acc_gain.shape[0])), col_indice.squeeze()] = -float('inf')
            rec_items_ = torch.cat(rec_items, dim=1)
            row = torch.arange(rec_items_.shape[0]).unsqueeze(1).expand(rec_items_.shape)
            row = row.flatten()
            col = rec_items_.flatten()
            values = torch.ones_like(row)
            rec_sparse_mat = sp.coo_matrix(
                (values, (row, col)), shape=[rec_items_.shape[0], item_genre_mat.shape[0]]).tocsr()
            rec_sparse_mat = rec_sparse_mat / rec_sparse_mat.sum(1)
            q = torch.sparse.mm(spy_sparse2torch_sparse(rec_sparse_mat), item_genre_mat)
            q = (1 - self.conf['beta']) * q + self.conf['beta'] * seq_user_genre_mat
        rec_items = torch.cat(rec_items, dim=1)
        acc_gains = torch.cat(acc_gains, dim=1)
        cali_gains = torch.cat(cali_gains, dim=1)
        return rec_items, acc_gains, cali_gains

    def get_acc_gain(self, pred):
        """
        Compute the gain of relevance
        """
        return torch.sigmoid(pred.cpu().detach())

    def get_cali_gain(self, user_genre_mat, item_genre_mat, q, k, balance):
        """
        Compute the gain of calibration
        """
        if q is None:
            prev_q = torch.full(user_genre_mat.shape, 1 / user_genre_mat.shape[1])
        else:
            prev_q = q
        prev_q = prev_q.unsqueeze(1)
        prev_q = prev_q.expand(prev_q.shape[0], item_genre_mat.shape[0], -1)
        next_q = (k * prev_q + item_genre_mat.unsqueeze(0).expand(prev_q.shape)) / (k+1)
        p = user_genre_mat.unsqueeze(1).expand(prev_q.shape)
        if k == 0:
            next_q = (1 - self.conf['beta']) * next_q + self.conf['beta'] * p
        score = p * torch.log(next_q / prev_q)
        cali_gain = score.sum(2)
        return cali_gain

    def get_metrics(self, metrics, grd, pred, user_genre_mat, seq_user_genre_mat, item_genre_mat, rec_items, topks):
        """
        Measure the results for each metrics
        """
        tmp = {'recall': {}, 'ndcg': {}, 'mcal': {}, 'smcal': {}}
        for topk in topks:
            col_indice = rec_items[:, :topk].contiguous()
            row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                     dtype=torch.long).view(-1, 1)
            is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
            tmp['recall'][topk] = get_recall(grd, is_hit)
            tmp['ndcg'][topk] = get_ndcg(grd, is_hit, topk)
            tmp['mcal'][topk] = get_mcal(col_indice, user_genre_mat, item_genre_mat, topk, beta=self.conf['beta'])
            tmp['smcal'][topk] = get_mcal(col_indice, seq_user_genre_mat, item_genre_mat, topk, beta=self.conf['beta'])

        for m, topk_res in tmp.items():
            for topk, res in topk_res.items():
                for i, x in enumerate(res):
                    metrics[m][topk][i] += x
        return metrics
