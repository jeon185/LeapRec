import pprint
import time

import click
import yaml

from model import LeapRec
from utils import *


@click.command()
@click.option('--data', type=str, default='ml-1m')
def main(data):
    conf = yaml.safe_load(open("config.yaml"))[data]
    conf['data'] = data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(conf)
    out_path = os.path.join(conf['out_path'], data)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(out_path, 'LeapRec_base.pt')

    # Data preparation
    dataset = data_partition(data, conf['data_path'])
    [user_train, user_valid, user_test, item_genre_mat] = dataset
    num_users, num_items = conf['num_users'], conf['num_items']
    num_batch = len(user_train) // conf['batch_size']
    total_len = 0.0
    for u in user_train:
        total_len += len(user_train[u])
    print(f'Average length of sequences: {total_len / len(user_train):.2f}')
    sampler = WarpSampler(user_train, num_users, num_items, batch_size=conf['batch_size'], max_len=conf['max_len'], n_workers=3)
    user_seq_valid = get_user_seq_valid(conf, num_users, user_train)
    user_seq_test = get_user_seq_test(conf, num_users, user_train, user_valid)

    # Model and optimizer definition
    decay_factor = [conf['alpha'] ** i for i in range(conf['max_len'])][::-1]
    decay_factor = torch.FloatTensor(decay_factor)
    model = LeapRec(num_users, num_items, conf, user_seq_valid, user_seq_test, item_genre_mat, decay_factor).to(device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], betas=(0.9, 0.98), weight_decay=conf['decay'])

    T = 0.0
    t0 = time.time()
    best_vld_ndcg = 0.

    for epoch in range(1, conf['epochs'] + 1):
        model.train()
        rec_loss_avg, kl_loss_avg = 0., 0.
        cur_instance_num = 0
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits, pos_kl, neg_kl = model(u, seq, pos, neg)
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            rec_loss = - (pos_logits[indices] - neg_logits[indices]).sigmoid().log().mean()
            kl_loss = - (pos_kl[indices] - neg_kl[indices]).sigmoid().log().mean()
            loss = rec_loss + conf['gamma'] * kl_loss
            loss.backward()
            optimizer.step()
            rec_loss_avg = moving_avg(rec_loss_avg, cur_instance_num, rec_loss.detach(), u.shape[0])
            kl_loss_avg = moving_avg(kl_loss_avg, cur_instance_num, kl_loss.detach(), u.shape[0])
            cur_instance_num += u.shape[0]

        print(f'Epoch: {epoch}, Rec Loss: {rec_loss_avg:.6f}, KL Loss: {kl_loss_avg:.6f}')

        if epoch % conf['test_interval'] == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, user_seq_test, num_users, num_items)
            t_valid = evaluate_valid(model, dataset, user_seq_valid, num_users, num_items)
            print('')
            print(f'Epoch: {epoch}, Time: {T:.1f}s, '
                  f'Valid (nDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
                  f'Test (nDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
            t0 = time.time()

            if t_valid[0] > best_vld_ndcg:
                best_vld_ndcg = t_valid[0]
                torch.save(model.state_dict(), model_path)

    sampler.close()

    print('============================ BEST ============================')
    model = LeapRec(num_users, num_items, conf, user_seq_valid, user_seq_test, item_genre_mat, decay_factor).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print('Evaluating', end='')
    t_test = evaluate(model, dataset, user_seq_test, num_users, num_items)
    t_valid = evaluate_valid(model, dataset, user_seq_valid, num_users, num_items)
    print('')
    print(f'Valid (nDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
          f'Test (nDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')


if __name__ == '__main__':
    main()
