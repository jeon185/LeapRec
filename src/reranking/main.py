import sys

import click
import yaml

sys.path.append('../')

from backbone.model import LeapRec
from backbone.utils import *
from algorithms import *
from utils import *


@click.command()
@click.option('--data', type=str, default='ml-1m')
@click.option('--algorithm', type=str, default='origin')
@click.option('--seed', type=int, default=0)
@click.option('--balance', type=float, default=0.1)
def main(data, algorithm, seed, balance):
    set_seed(seed)
    conf = yaml.safe_load(open("config.yaml"))[data]
    conf['dataset'] = data
    dataset = Datasets(conf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    out_path = os.path.join(conf['out_path'], data)
    model_path = os.path.join(out_path, 'LeapRec_base.pt')
    [user_train, user_valid, user_test, item_genre_mat] = data_partition(data, conf['data_path'])
    num_users, num_items = conf['num_users'], conf['num_items']
    user_seq_valid = get_user_seq_valid(conf, num_users, user_train)
    user_seq_test = get_user_seq_test(conf, num_users, user_train, user_valid)
    decay_factor = [conf['alpha'] ** i for i in range(conf['max_len'])][::-1]
    decay_factor = torch.FloatTensor(decay_factor)
    model = LeapRec(num_users, num_items, conf, user_seq_valid, user_seq_test, item_genre_mat, decay_factor).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(conf['device'])

    if algorithm == 'origin':
        algorithm = Origin()
        algorithm.get_dataset(model, dataset.test_loader,
                              dataset.user_genre_mat, dataset.seq_user_genre_mat,
                              dataset.item_genre_mat, conf)
        algorithm.evaluate()

    elif algorithm == 'leaprec':
        algorithm = LeapRecRerank(balance=balance)
        algorithm.get_dataset(model, dataset.test_loader,
                              dataset.user_genre_mat, dataset.seq_user_genre_mat,
                              dataset.item_genre_mat, conf)
        algorithm.evaluate()


if __name__ == '__main__':
    main()
