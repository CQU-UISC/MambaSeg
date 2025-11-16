import yaml
import argparse
from utils.trainer import Trainer
from utils.func import random_seed
from utils.func import modify_log_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='configs/DDD17.yaml',
                        help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Random seed
    random_seed(cfg['SEED_NUM'])
    # Set log_dir
    cfg['TRAIN']['log_dir'] = modify_log_dir(cfg)

    # print(cfg)
    # Initialize the trainer
    trainer = Trainer(cfg)
    trainer.train()
