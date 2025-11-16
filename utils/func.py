import torch
import os
import random
import numpy as np


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def random_seed(num):
    np.random.seed(num)
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def modify_log_dir(cfg):
    # 根据dataset和model_name等信息修改log_dir
    # 格式为：log_dir/dataset/data_type/exp_name
    # 例如：log_dir/DDD17/fusion/exp_name
    dataset = cfg['DATASET']['name']
    data_type = cfg['DATASET']['data_type']
    exp_name = cfg['MODEL']['name']
    fuse = cfg['MODEL']['fuse']
    ev_rep = cfg['DATASET']['event_representation']
    batch_size = cfg['TRAIN']['batch_size']
    lr_scheduler = cfg['TRAIN']['lr_scheduler']
    exp_name = exp_name + '_' + fuse + '_' + ev_rep + '_' + str(batch_size)+ '_' + lr_scheduler
    base_dir = 'log'
    log_dir = os.path.join(base_dir, dataset, data_type, exp_name)

    return log_dir